"""Backfill ML events for a user from 5 listing endpoints.

Called:
- Manually via POST /api/v2/escalar/backfill-notices  (days=30)
- Automatically after OAuth-connect             (days=30)
- APScheduler daily job                        (days=1, catch-up for webhook gaps)

Normalizes every fetched record through ml_normalize.normalize_event and
bulk-upserts into ml_notices (same UNIQUE(user_id, notice_id) constraint).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_normalize
from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
PER_ENDPOINT_LIMIT = 50
RATE_SLEEP = 0.05


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _get_ml(http: httpx.AsyncClient, access_token: str, path: str) -> Any:
    try:
        r = await http.get(
            f"{ML_API_BASE}{path}" if path.startswith("/") else path,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20.0,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("ML GET %s failed: %s", path, err)
        return None
    if r.status_code != 200:
        log.warning("ML GET %s → %s: %s", path, r.status_code, r.text[:200])
        return None
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return None


async def _get_ml_user_id(pool: asyncpg.Pool, user_id: int) -> int | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ml_user_id FROM ml_user_tokens WHERE user_id = $1", user_id,
        )
    return row["ml_user_id"] if row else None


# ── Source fetchers ───────────────────────────────────────────────────────────

async def _fetch_orders(http: httpx.AsyncClient, token: str, ml_user_id: int, since: datetime) -> list[dict]:
    iso = since.strftime("%Y-%m-%dT%H:%M:%S.000-03:00")
    data = await _get_ml(
        http, token,
        f"/orders/search?seller={ml_user_id}&order.date_created.from={iso}&limit={PER_ENDPOINT_LIMIT}",
    )
    return (data or {}).get("results") or []


async def _fetch_questions(http: httpx.AsyncClient, token: str) -> list[dict]:
    # ML /my/received_questions/search accepts ONE status at a time — fetch both
    # UNANSWERED and ANSWERED separately and merge (+ sort by date_created DESC).
    unanswered, answered = await asyncio.gather(
        _get_ml(
            http, token,
            f"/my/received_questions/search?status=UNANSWERED"
            f"&sort_fields=date_created&sort_types=DESC"
            f"&api_version=4&limit={PER_ENDPOINT_LIMIT}",
        ),
        _get_ml(
            http, token,
            f"/my/received_questions/search?status=ANSWERED"
            f"&sort_fields=date_created&sort_types=DESC"
            f"&api_version=4&limit={PER_ENDPOINT_LIMIT}",
        ),
    )
    merged: list[dict] = []
    for bucket in (unanswered, answered):
        if bucket and isinstance(bucket, dict):
            merged.extend(bucket.get("questions") or [])
    return merged


async def _enrich_claim_with_returns(http: httpx.AsyncClient, token: str, claim: dict) -> dict:
    """Fetch full claim + returns + triage reviews. Returns enriched copy."""
    claim_id = claim.get("id")
    if not claim_id:
        return claim
    full = await _get_ml(http, token, f"/post-purchase/v1/claims/{claim_id}") or {}
    merged = {**claim, **full}

    related = merged.get("related_entities") or []
    has_return = any(
        (isinstance(e, str) and e == "returns")
        or (isinstance(e, dict) and (e.get("type") == "returns" or e.get("name") == "returns"))
        for e in related
    )
    if not has_return:
        return merged

    returns_data = await _get_ml(http, token, f"/post-purchase/v2/claims/{claim_id}/returns")
    if not returns_data:
        return merged
    returns_list = returns_data.get("data") if isinstance(returns_data, dict) else returns_data
    if isinstance(returns_list, dict):
        returns_list = [returns_list]
    if not isinstance(returns_list, list):
        returns_list = []

    for ret in returns_list:
        ret_related = ret.get("related_entities") or []
        has_reviews = any(
            (isinstance(e, str) and e == "reviews")
            or (isinstance(e, dict) and (e.get("type") == "reviews" or e.get("name") == "reviews"))
            for e in ret_related
        )
        if has_reviews and ret.get("id"):
            rev = await _get_ml(http, token, f"/post-purchase/v1/returns/{ret['id']}/reviews")
            if rev:
                ret["reviews"] = rev.get("reviews") if isinstance(rev, dict) else rev

    merged["returns"] = returns_list
    return merged


async def _fetch_claims(http: httpx.AsyncClient, token: str) -> list[dict]:
    data = await _get_ml(
        http, token,
        f"/post-purchase/v1/claims/search?player_role=seller&limit={PER_ENDPOINT_LIMIT}",
    )
    claims = (data or {}).get("data") or (data or {}).get("results") or []
    if not claims:
        return []
    # Enrich each claim with full payload + returns + reviews (batched by 10).
    enriched: list[dict] = []
    for i in range(0, len(claims), 10):
        batch = claims[i:i + 10]
        results = await asyncio.gather(
            *[_enrich_claim_with_returns(http, token, c) for c in batch],
            return_exceptions=True,
        )
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                enriched.append(batch[idx])  # keep bare row on error
            else:
                enriched.append(r)
    return enriched


async def _fetch_paused_closed_items(http: httpx.AsyncClient, token: str, ml_user_id: int) -> list[dict]:
    async def _ids_for_status(status: str) -> list[str]:
        data = await _get_ml(
            http, token,
            f"/users/{ml_user_id}/items/search?status={status}&limit={PER_ENDPOINT_LIMIT}",
        )
        return (data or {}).get("results") or []

    ids_paused, ids_closed = await asyncio.gather(
        _ids_for_status("paused"),
        _ids_for_status("closed"),
    )
    ids = list(dict.fromkeys([*ids_paused, *ids_closed]))[:100]
    if not ids:
        return []

    details: list[dict] = []
    # Batch /items?ids=... (up to 20 per call)
    for i in range(0, len(ids), 20):
        batch = ids[i:i + 20]
        data = await _get_ml(
            http, token,
            f"/items?ids={','.join(batch)}&attributes=id,status,sub_status,title,permalink,last_updated",
        )
        if isinstance(data, list):
            for r in data:
                if r and r.get("code") == 200 and r.get("body"):
                    details.append(r["body"])
    return details


async def _fetch_messages(http: httpx.AsyncClient, token: str, ml_user_id: int) -> list[dict]:
    data = await _get_ml(
        http, token,
        f"/messages/actions?seller_id={ml_user_id}&limit={PER_ENDPOINT_LIMIT}",
    )
    if not data:
        return []
    # API shape varies — accept either a list or {results: [...]}.
    if isinstance(data, list):
        return data
    return data.get("results") or data.get("messages") or []


# ── Upsert ────────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO ml_notices
  (user_id, notice_id, label, description, from_date, tags, actions, raw,
   topic, resource, updated_at)
VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, NOW())
ON CONFLICT (user_id, notice_id) DO UPDATE SET
  label = EXCLUDED.label,
  description = EXCLUDED.description,
  from_date = EXCLUDED.from_date,
  tags = EXCLUDED.tags,
  actions = EXCLUDED.actions,
  raw = EXCLUDED.raw,
  topic = EXCLUDED.topic,
  resource = EXCLUDED.resource,
  updated_at = NOW()
"""


async def _enrich_order_with_margin(
    user_id: int, enriched: dict, period_months: int = 3,
    pool: Optional[asyncpg.Pool] = None,
) -> None:
    """Injects `_margin` (apply_hypothetical_price от sale_price) в order payload.

    Mutates `enriched` in-place. Без модификации БД. Если cache empty или
    нет item_id / unit_price — просто не инжектит, normalize-ветка покажет
    «Margem indisponível». Для SMART/auto-activated promotions accept этот
    block не нужен — это только для orders_v2 уведомлений.

    Также инжектит `_breakeven` — state break-even tracker'а после этой
    продажи (cumulative variable margin + target + breakeven_reached). Это
    показывается в TG normalize как "📈 Прогресс месяца".

    Acquires its own connections from `pool` and releases each before calling
    subservices (breakeven / inventory / paused-with-stock). Не принимает
    caller-owned `conn` — иначе под concurrent webhook load 5+ хэндлеров
    держат по одному коннекту и одновременно делают nested pool.acquire(),
    что исчерпывает пул и навечно блокирует scheduler-задачи.
    """
    if pool is None:
        return
    from . import ml_item_margin as ml_margin_svc
    from . import ml_breakeven as breakeven_svc

    items_arr = enriched.get("order_items") or enriched.get("items") or []
    if not items_arr:
        return
    first = items_arr[0] if isinstance(items_arr[0], dict) else None
    if not first:
        return
    inner = first.get("item") if isinstance(first.get("item"), dict) else first
    item_id = ""
    if isinstance(inner, dict):
        item_id = str(inner.get("id") or inner.get("mlb") or "").strip().upper()
    if not item_id:
        item_id = str(first.get("mlb") or "").strip().upper()
    try:
        sale_price = float(first.get("unit_price") or 0.0)
    except (TypeError, ValueError):
        sale_price = 0.0
    if not item_id or sale_price <= 0:
        # Even if sale_price=0 (e.g., cancelled order), inject cancellation
        # stats when status is cancelled/invalid. Webhook handler at
        # escalar.py:3501 also tries, but its debug-level exception handler
        # swallowed the error silently and backfill path never ran this.
        status_raw = str(enriched.get("status") or "").lower()
        if item_id and status_raw in ("cancelled", "invalid"):
            try:
                from . import ml_orders as _ml_orders_cancel
                cancel_stats = await _ml_orders_cancel.get_cancellation_stats(
                    pool, user_id, item_id,
                )
                if cancel_stats:
                    enriched["_cancellation_stats"] = cancel_stats
            except Exception as err:  # noqa: BLE001
                log.warning("cancel stats in _enrich failed %s: %s", item_id, err)
        return

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT payload, computed_at
              FROM ml_item_margin_cache
             WHERE user_id = $1 AND item_id = $2 AND period_months = $3
            """,
            user_id, item_id, period_months,
        )
    if not row:
        return
    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:  # noqa: BLE001
            return
    if not isinstance(payload, dict):
        return
    payload["computed_at"] = row["computed_at"].isoformat() if row["computed_at"] else None
    try:
        recomputed = ml_margin_svc.apply_hypothetical_price(payload, sale_price)
        enriched["_margin"] = recomputed
    except Exception as err:  # noqa: BLE001
        log.debug("margin recompute failed for %s: %s", item_id, err)
        return

    # Negative-margin snooze check — если seller добавил этот SKU в snooze
    # list (через TG button «🤐 Não notificar este SKU»), не показываем
    # prejuízo-alert в TG. ml_normalize прочитает enriched._negative_margin_snoozed
    # и пропустит alert-блок + не добавит кнопки.
    try:
        seller_sku_raw = ""
        if isinstance(inner, dict):
            seller_sku_raw = str(inner.get("seller_sku") or "").strip()
        if not seller_sku_raw:
            seller_sku_raw = str(first.get("seller_sku") or "").strip()
        if seller_sku_raw:
            async with pool.acquire() as conn_snz:
                snz_row = await conn_snz.fetchrow(
                    "SELECT data_value FROM user_data "
                    "WHERE user_id = $1 AND data_key = 'f2_neg_margin_snoozed_skus' LIMIT 1",
                    user_id,
                )
            snoozed: list[str] = []
            if snz_row:
                _val = snz_row["data_value"]
                if isinstance(_val, list):
                    snoozed = [str(x).strip() for x in _val if x]
                elif isinstance(_val, str):
                    try:
                        _parsed = json.loads(_val)
                        if isinstance(_parsed, list):
                            snoozed = [str(x).strip() for x in _parsed if x]
                    except Exception:  # noqa: BLE001
                        pass
            if seller_sku_raw in snoozed:
                enriched["_negative_margin_snoozed"] = True
    except Exception as err:  # noqa: BLE001
        log.debug("negative margin snooze check failed: %s", err)

    # ── Cancellation stats per-SKU ────────────────────────────────────────
    # Если заказ cancelled/invalid — подтянуть % отмен по этому MLB из
    # ml_user_orders. Делаем тут (а не только в escalar.py:3501) потому что:
    # 1) webhook handler ловит exception молча (был debug-level)
    # 2) backfill path (_upsert_batch) вообще не инжектит stats
    # 3) _enrich_order_with_margin вызывается из ОБОИХ путей (webhook + backfill)
    status_for_cancel = str(enriched.get("status") or "").lower()
    if item_id and status_for_cancel in ("cancelled", "invalid"):
        try:
            from . import ml_orders as _ml_orders_cancel
            cancel_stats = await _ml_orders_cancel.get_cancellation_stats(
                pool, user_id, item_id,
            )
            if cancel_stats:
                enriched["_cancellation_stats"] = cancel_stats
        except Exception as err:  # noqa: BLE001
            log.warning("cancel stats inject %s: %s", item_id, err)

    # ── Velocity adjustment ───────────────────────────────────────────────
    # Cached margin uses vendas CSV для units_sold count, который юзер
    # обновляет вручную раз в N дней. Когда продажи ускорились (например
    # +20×) — vendas-based count устарел и fixed_overhead_per_unit
    # завышен в 20×. ml_user_orders (API source, обновляется автоматически)
    # имеет правильную velocity. Если API-count существенно выше cached —
    # масштабируем fixed_overhead_per_unit и пересчитываем net.
    try:
        from datetime import datetime as _dt2, timedelta as _td, timezone as _tz
        from . import ml_orders as _ml_orders_svc
        cached_units = int(payload.get("units_sold") or 0)
        if cached_units > 0:
            period_months = int(payload.get("period_months") or 3)
            end = _dt2.now(_tz.utc)
            start = end - _td(days=30 * period_months)
            win = await _ml_orders_svc.get_orders_for_window(
                pool, user_id, item_id, start, end,
            )
            api_units = int(win.get("units") or 0)
            # Only scale DOWN — никогда не увеличиваем overhead/unit
            # на случай если ml_user_orders ещё не догнал backfill.
            if api_units > cached_units * 1.5:  # 50%+ выше cached
                ratio = cached_units / api_units
                unit_d = recomputed.get("unit") or {}
                old_fixed = unit_d.get("fixed_overhead_per_unit")
                profit_var = unit_d.get("profit_variable")
                price_b = unit_d.get("current_price") or sale_price
                if old_fixed is not None and profit_var is not None and price_b:
                    new_fixed = float(old_fixed) * ratio
                    new_net = float(profit_var) - new_fixed
                    unit_d["fixed_overhead_per_unit"] = round(new_fixed, 2)
                    unit_d["profit_net_per_unit"] = round(new_net, 2)
                    unit_d["margin_net_pct"] = round(new_net / float(price_b) * 100, 1)
                    recomputed["velocity_adjustment"] = {
                        "cached_units_3m": cached_units,
                        "api_units_3m": api_units,
                        "ratio": round(ratio, 4),
                        "source": "ml_user_orders",
                    }
                    recomputed["unit"] = unit_d
                    enriched["_margin"] = recomputed
    except Exception as err:  # noqa: BLE001
        log.debug("velocity adjustment failed for %s: %s", item_id, err)

    # ── Break-even explainer (когда net < 0) ─────────────────────────────
    # «Чтобы выйти в плюс по проекту»: считаем PROJECT-level (а не per-item),
    # так как user явно требовал «по проекту всего»:
    #   - sum(overhead_total) across all items in project = project monthly fixed
    #   - weighted avg per-sale margin = SUM(pv × units) / SUM(units) across project
    #   - sales needed/мес = ceil(project_monthly_fixed / weighted_avg_margin)
    #   - sales actual this calendar month = sum units across all project items
    #     in ml_user_orders где status not cancelled
    try:
        unit_d = (recomputed.get("unit") or {})
        net_pu = unit_d.get("profit_net_per_unit")
        pv_pu = unit_d.get("profit_variable")
        project_norm = (project or "").upper().strip() if 'project' in dir() else ""
        # Re-derive project here (block runs before main project resolution below).
        proj_from_payload = (
            recomputed.get("project") or payload.get("project") or ""
        )
        project_for_be = str(proj_from_payload).upper().strip()
        if (
            net_pu is not None and float(net_pu) < 0
            and pv_pu is not None and float(pv_pu) > 0
            and project_for_be
        ):
            import math as _math
            async with pool.acquire() as conn_be:
                # PROJECT-level: agregate ТОЛЬКО user-entered manual fixed
                # costs (project config UI). Computed shares (publi из ads,
                # armaz, fulfillment, aluguel-empresa) — НЕ включаем потому
                # что:
                # - реклама = инвестиция, не фикс расход
                # - fulfillment скейлится с продажами, не фикс
                # - aluguel-empresa pro-rata из company-wide, не project
                # User explicitly: «полагаемся только на фикс расходы что мы
                # указали в проекте, от них отталкиваемся». Так что берём
                # manual_fc_user_total (sum of salaries + utilities + software
                # + outros + aluguel + armazenagem categories из project
                # config). 0 → informational note вместо breakeven calc.
                project_row = await conn_be.fetchrow(
                    """
                    SELECT COUNT(*) AS n_items,
                           MAX((payload->'unit'->>'manual_fc_user_total')::float)
                             AS manual_fc_monthly,
                           COALESCE(SUM((payload->>'units_sold')::int), 0) AS units_sum,
                           COALESCE(SUM(
                             (payload->'unit'->>'profit_variable')::float
                             * (payload->>'units_sold')::int
                           ), 0) AS weighted_pv_sum,
                           MAX((payload->>'period_months')::int) AS period_m,
                           ARRAY_AGG(item_id) AS item_ids
                      FROM ml_item_margin_cache
                     WHERE user_id = $1
                       AND UPPER(TRIM(payload->>'project')) = $2
                       AND (payload->>'ok')::boolean = TRUE
                    """,
                    user_id, project_for_be,
                )
                manual_fc_monthly = float((project_row or {}).get("manual_fc_monthly") or 0)
                units_sum = int((project_row or {}).get("units_sum") or 0)
                weighted_pv = float((project_row or {}).get("weighted_pv_sum") or 0)
                period_m = int((project_row or {}).get("period_m") or 3)
                proj_item_ids = list((project_row or {}).get("item_ids") or [])
                n_items = int((project_row or {}).get("n_items") or 0)

                # User-configured rental (compensation_mode='rental'). UI:
                # «RATE USD: 800» + «PERIOD: quarter» → rental.rate_usd +
                # rental.period в project config. Берём ИЗ НАСТРОЕК (rate_usd),
                # не из payments. Конвертим в BRL/мес по последнему paid
                # rate_brl, fallback 5.3.
                rental_monthly_brl = 0.0
                rental_label = ""
                proj_cfg_row = await conn_be.fetchrow(
                    "SELECT data_value FROM user_data "
                    "WHERE user_id = $1 AND data_key = 'projects' LIMIT 1",
                    user_id,
                )
                if proj_cfg_row:
                    proj_cfg = proj_cfg_row["data_value"]
                    if isinstance(proj_cfg, str):
                        try:
                            proj_cfg = json.loads(proj_cfg)
                        except Exception:  # noqa: BLE001
                            proj_cfg = {}
                    if isinstance(proj_cfg, dict):
                        for _pname, _pdata in proj_cfg.items():
                            if (_pname or "").upper().strip() != project_for_be:
                                continue
                            if not isinstance(_pdata, dict):
                                continue
                            rental_cfg = _pdata.get("rental") or {}
                            if not isinstance(rental_cfg, dict):
                                break
                            try:
                                rate_usd = float(rental_cfg.get("rate_usd") or 0)
                            except (TypeError, ValueError):
                                rate_usd = 0.0
                            if rate_usd <= 0:
                                break
                            period_str = str(rental_cfg.get("period") or "month").lower()
                            period_divisor = {
                                "month": 1, "monthly": 1,
                                "quarter": 3, "quarterly": 3,
                                "semester": 6,
                                "year": 12, "yearly": 12, "annual": 12,
                            }.get(period_str, 1)
                            # Latest paid rate_brl from payments (для конверсии).
                            rate_brl = 5.3
                            paid_rates: list[tuple[str, float]] = []
                            for _p in (rental_cfg.get("payments") or []):
                                if (
                                    isinstance(_p, dict)
                                    and (_p.get("status") or "").lower() == "paid"
                                    and _p.get("rate_brl") is not None
                                ):
                                    try:
                                        paid_rates.append((
                                            str(_p.get("date") or ""),
                                            float(_p.get("rate_brl")),
                                        ))
                                    except (TypeError, ValueError):
                                        pass
                            if paid_rates:
                                paid_rates.sort(key=lambda x: x[0], reverse=True)
                                rate_brl = paid_rates[0][1]
                            rental_monthly_brl = (rate_usd * rate_brl) / period_divisor
                            rental_label = (
                                f"US$ {int(rate_usd)}/{period_str} × R$ {rate_brl:.2f}"
                            )
                            break

                project_monthly_fixed = manual_fc_monthly + rental_monthly_brl
                avg_pv_per_sale = weighted_pv / units_sum if units_sum > 0 else float(pv_pu)

                if avg_pv_per_sale <= 0 or not proj_item_ids:
                    raise RuntimeError("project breakeven inputs not viable")

                sales_needed = (
                    int(_math.ceil(project_monthly_fixed / avg_pv_per_sale))
                    if project_monthly_fixed > 0 else 0
                )

                # Actual sales this calendar month across ALL project items.
                actual_row = await conn_be.fetchrow(
                    """
                    SELECT COALESCE(SUM((items_elem->>'quantity')::int), 0) AS units
                      FROM ml_user_orders, jsonb_array_elements(items) items_elem
                     WHERE user_id = $1
                       AND status NOT IN ('cancelled','invalid')
                       AND items_elem->>'mlb' = ANY($2::text[])
                       AND date_created >= date_trunc('month', NOW())
                    """,
                    user_id, proj_item_ids,
                )
            actual_mtd = int((actual_row or {}).get("units") or 0)
            month_so_far_brl = round(actual_mtd * avg_pv_per_sale, 2)
            coverage_pct = (
                round((actual_mtd / sales_needed) * 100, 1)
                if sales_needed > 0 else None
            )
            enriched["_breakeven_explainer"] = {
                "scope": "project",
                "project": project_for_be,
                "n_items": n_items,
                "monthly_fixed_brl": round(project_monthly_fixed, 2),
                "manual_fc_monthly_brl": round(manual_fc_monthly, 2),
                "rental_monthly_brl": round(rental_monthly_brl, 2),
                "rental_label": rental_label,
                "profit_variable_per_sale": round(avg_pv_per_sale, 2),
                "sales_needed_per_month": sales_needed,
                "sales_actual_mtd": actual_mtd,
                "coverage_pct": coverage_pct,
                "gap_units": max(0, sales_needed - actual_mtd),
                "month_so_far_brl": month_so_far_brl,
                "monthly_target_brl": round(project_monthly_fixed, 2),
                "manual_fc_configured": project_monthly_fixed > 0,
            }
    except Exception as err:  # noqa: BLE001
        log.debug("breakeven explainer failed for %s: %s", item_id, err)

    # Break-even tracker: добавить эту продажу в cumulative проекта/месяца.
    # Project определяется из margin payload (тот же что в /escalar/products).
    project = recomputed.get("project") or payload.get("project")
    if not project:
        return
    unit = recomputed.get("unit") or {}
    profit_variable = unit.get("profit_variable")
    qty = 1
    try:
        qty = int(first.get("quantity") or 1)
    except (TypeError, ValueError):
        qty = 1
    if profit_variable is None:
        return
    try:
        total_profit_var = float(profit_variable) * qty
    except (TypeError, ValueError):
        return

    # parse sale date (BRT) для year_month identification.
    from datetime import datetime as _dt
    sale_dt: Optional[_dt] = None
    raw_dt = enriched.get("date_created") or enriched.get("last_updated")
    if raw_dt:
        try:
            sale_dt = _dt.fromisoformat(str(raw_dt).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            sale_dt = None

    try:
        await breakeven_svc.ensure_schema(pool)
        order_id_for_dedup = str(enriched.get("id") or "").strip() or None
        be_state = await breakeven_svc.add_sale_and_check_breakeven(
            pool, user_id, str(project), total_profit_var, sale_dt,
            order_id=order_id_for_dedup,
        )
        if be_state:
            enriched["_breakeven"] = be_state
            # Workaround для устаревшего ml_user_items.available_quantity:
            # cache синкается раз в 24ч, поэтому 2 продажи подряд показывают
            # одинаковый stock. Декрементируем cache на qty при first-time
            # обработке order'а (deduplicated=False — видим этот order
            # впервые в breakeven_sale_log). Live ML API заменит этот
            # workaround когда мы интегрируем /full-fulfillment-stock.
            if not be_state.get("deduplicated") and order_id_for_dedup and qty > 0:
                try:
                    async with pool.acquire() as inv_conn:
                        await inv_conn.execute(
                            """
                            UPDATE ml_user_items
                               SET available_quantity = GREATEST(
                                     COALESCE(available_quantity, 0) - $3, 0
                                   )
                             WHERE user_id = $1 AND item_id = $2
                            """,
                            user_id, item_id, qty,
                        )
                except Exception as err:  # noqa: BLE001
                    log.debug("stock decrement failed item=%s: %s", item_id, err)
    except Exception as err:  # noqa: BLE001
        log.debug("breakeven update failed for project=%s: %s", project, err)

    # Inventory forecast — стoк + 14d скорость → сколько дней хватит.
    # Skip silently если service не доступен / item не в ml_user_items.
    # Если у order есть variation_id — берём stock per-variation (точнее
    # для multi-variant товаров).
    try:
        from . import ml_inventory_forecast as inv_svc
        await inv_svc.ensure_schema(pool)
        var_id_for_inv: Optional[str] = None
        if isinstance(inner, dict):
            var_id_for_inv = str(inner.get("variation_id") or "").strip() or None
        if not var_id_for_inv:
            var_id_for_inv = str(first.get("variation_id") or "").strip() or None
        snap = await inv_svc.get_inventory_snapshot(
            pool, user_id, item_id, variation_id=var_id_for_inv,
        )
        if snap:
            enriched["_inventory"] = snap
    except Exception as err:  # noqa: BLE001
        log.debug("inventory snapshot failed for item=%s: %s", item_id, err)

    # Paused-with-stock — товары которые юзер либо сам поставил на паузу,
    # либо ML модерация, но stock реально есть. Это «мёртвый Full капитал»
    # без трафика.
    #
    # User feedback: list повторялся в КАЖДОЙ продаже + showed items
    # которые user уже активировал (cache stale). Два fix'а:
    #   1. Live batch verify статуса через GET /items?ids=... (top-5
    #      candidates → keep only тех кто реально paused в ML).
    #   2. Per-day dedup через paused_alert_log — один item не показывается
    #      повторно в течение 24h.
    try:
        import os as _os_pws
        _app_base_pws = _os_pws.environ.get("APP_BASE_URL", "https://app.lsprofit.app").rstrip("/")
        # ensure dedup log table
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paused_alert_log (
                  user_id INTEGER NOT NULL,
                  item_id TEXT NOT NULL,
                  alerted_at TIMESTAMPTZ DEFAULT NOW(),
                  PRIMARY KEY (user_id, item_id)
                );
                """
            )
            paused_rows = await conn.fetch(
                """
                SELECT i.item_id, i.available_quantity, i.sold_quantity, i.title
                  FROM ml_user_items i
                  LEFT JOIN paused_alert_log p
                    ON p.user_id = i.user_id AND p.item_id = i.item_id
                 WHERE i.user_id = $1
                   AND i.status = 'paused'
                   AND i.available_quantity > 0
                   AND (p.alerted_at IS NULL
                        OR p.alerted_at < NOW() - INTERVAL '24 hours')
                 ORDER BY i.available_quantity DESC, i.sold_quantity DESC
                 LIMIT 5
                """,
                user_id,
            )
        if paused_rows:
            # Live batch verify status — cache может быть устаревшим (24h
            # sync), пользователь мог уже активировать через UI.
            candidate_ids = [str(r["item_id"]) for r in paused_rows]
            confirmed_paused: set[str] = set()
            try:
                from . import ml_oauth as _oauth
                token, _exp, _ = await _oauth.get_valid_access_token(pool, user_id)
                import httpx
                async with httpx.AsyncClient() as _http:
                    ids_param = ",".join(candidate_ids[:5])
                    url = (
                        f"https://api.mercadolibre.com/items"
                        f"?ids={ids_param}&attributes=id,status"
                    )
                    r = await _http.get(
                        url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                    if r.status_code < 400:
                        live_data = r.json() or []
                        live_active_ids: set[str] = set()
                        for entry in live_data:
                            body = (entry or {}).get("body") or entry
                            if not isinstance(body, dict):
                                continue
                            iid = str(body.get("id") or "")
                            if not iid:
                                continue
                            live_status = str(body.get("status") or "").lower()
                            if live_status == "paused":
                                confirmed_paused.add(iid.upper())
                            elif live_status == "active":
                                live_active_ids.add(iid.upper())
                        # Sync cache: items которые seller уже активировал
                        # → update ml_user_items.status чтобы будущие
                        # запросы не извлекали их.
                        if live_active_ids:
                            async with pool.acquire() as _conn2:
                                await _conn2.execute(
                                    """
                                    UPDATE ml_user_items
                                       SET status = 'active'
                                     WHERE user_id = $1
                                       AND item_id = ANY($2::text[])
                                    """,
                                    user_id, list(live_active_ids),
                                )
            except Exception as err:  # noqa: BLE001
                log.debug("paused-with-stock live verify failed: %s", err)
                # Fallback — без live check используем cache statuses
                confirmed_paused = {str(r["item_id"]).upper() for r in paused_rows}

            paused_list: list[dict] = []
            for pr in paused_rows:
                pid = str(pr["item_id"]).upper()
                if pid not in confirmed_paused:
                    continue
                paused_list.append({
                    "item_id": pid,
                    "stock": int(pr["available_quantity"] or 0),
                    "sold": int(pr["sold_quantity"] or 0),
                    "title": pr["title"] or "",
                    "activate_url": f"{_app_base_pws}/escalar/activate/{pid}",
                })
                if len(paused_list) >= 3:
                    break

            if paused_list:
                # Mark as alerted — UPSERT alerted_at = NOW() для каждого
                # показанного item, чтобы 24h dedup сработал.
                async with pool.acquire() as conn:
                    for p in paused_list:
                        await conn.execute(
                            """
                            INSERT INTO paused_alert_log (user_id, item_id, alerted_at)
                            VALUES ($1, $2, NOW())
                            ON CONFLICT (user_id, item_id) DO UPDATE
                              SET alerted_at = NOW()
                            """,
                            user_id, p["item_id"],
                        )
                enriched["_paused_with_stock"] = paused_list
    except Exception as err:  # noqa: BLE001
        log.debug("paused-with-stock enrich failed: %s", err)


async def _upsert_batch(
    conn: asyncpg.Connection,
    user_id: int,
    items: list[tuple[str, str | None, dict]],
    pool: Optional[asyncpg.Pool] = None,
) -> int:
    saved = 0
    for topic, resource, enriched in items:
        # For orders — pre-enrich with cached unit margin (apply_hypothetical_price
        # at sale_price) so normalize-ветка orders_v2 рисует profit/breakdown.
        # Enrich manages its own pool acquires; не передаём наш `conn`, иначе
        # nested acquire-while-holding-conn исчерпывает пул под concurrent load.
        if topic in ("orders_v2", "orders"):
            try:
                await _enrich_order_with_margin(user_id, enriched, pool=pool)
            except Exception as err:  # noqa: BLE001
                log.debug("enrich order margin failed: %s", err)
        try:
            notice = ml_normalize.normalize_event(topic, resource, enriched)
        except Exception as err:  # noqa: BLE001
            log.warning("normalize failed for %s: %s", topic, err)
            continue
        try:
            from . import ml_notices as _ml_notices_svc
            await conn.execute(
                _UPSERT_SQL,
                user_id,
                notice["notice_id"],
                notice.get("label"),
                notice.get("description"),
                _ml_notices_svc._coerce_to_datetime(notice.get("from_date")),
                json.dumps(notice.get("tags") or []),
                json.dumps(notice.get("actions") or []),
                json.dumps(notice.get("raw") or {}),
                notice.get("topic"),
                notice.get("resource"),
            )
            saved += 1
        except Exception as err:  # noqa: BLE001
            log.warning("upsert failed for %s: %s", notice.get("notice_id"), err)
    return saved


# ── Entry points ──────────────────────────────────────────────────────────────

async def backfill_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
    days: int = 30,
) -> dict[str, int]:
    """Fetch 5 source endpoints, normalize, bulk-upsert. Returns counts."""
    try:
        access_token, _expires, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("backfill user %s: no valid token: %s", user_id, err)
        return {"user_id": user_id, "fetched": 0, "saved": 0}

    ml_user_id = await _get_ml_user_id(pool, user_id)
    if not ml_user_id:
        return {"user_id": user_id, "fetched": 0, "saved": 0}

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # All 5 in parallel
    orders, questions, claims, items_detail, messages = await asyncio.gather(
        _fetch_orders(http, access_token, ml_user_id, since),
        _fetch_questions(http, access_token),
        _fetch_claims(http, access_token),
        _fetch_paused_closed_items(http, access_token, ml_user_id),
        _fetch_messages(http, access_token, ml_user_id),
        return_exceptions=False,
    )

    batch: list[tuple[str, str | None, dict]] = []
    for o in orders:
        oid = o.get("id")
        if oid:
            batch.append(("orders_v2", f"/orders/{oid}", o))
    for q in questions:
        qid = q.get("id")
        if qid:
            batch.append(("questions", f"/questions/{qid}", q))
    for c in claims:
        cid = c.get("id") or c.get("resource_id")
        if cid:
            batch.append(("claims", f"/post-purchase/v1/claims/{cid}", c))
    for it in items_detail:
        iid = it.get("id")
        if iid:
            batch.append(("items", f"/items/{iid}", it))
    for m in messages:
        mid = m.get("id") or m.get("message_id") or m.get("pack_id")
        if mid:
            batch.append(("messages", f"/messages/{mid}", m))

    fetched = len(batch)
    saved = 0
    if batch:
        async with pool.acquire() as conn:
            saved = await _upsert_batch(conn, user_id, batch, pool=pool)

    log.info("backfill user=%s days=%s fetched=%s saved=%s", user_id, days, fetched, saved)
    return {"user_id": user_id, "fetched": fetched, "saved": saved}


async def backfill_all_users(pool: asyncpg.Pool, days: int = 1) -> dict[str, int]:
    """APScheduler daily catch-up. Runs backfill_user for every known ML-connected user."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
        )
    totals = {"users": 0, "fetched": 0, "saved": 0}
    if not rows:
        return totals

    async with httpx.AsyncClient() as http:
        for r in rows:
            try:
                res = await backfill_user(pool, http, r["user_id"], days=days)
                totals["users"] += 1
                totals["fetched"] += res["fetched"]
                totals["saved"] += res["saved"]
            except Exception as err:  # noqa: BLE001
                log.exception("daily backfill user %s failed: %s", r["user_id"], err)
            await asyncio.sleep(RATE_SLEEP)
    return totals
