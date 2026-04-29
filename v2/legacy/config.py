"""
LATAMSELLERS — Configuration for financial accounting system
"""
import copy
import functools
import json
import os
from pathlib import Path

# === PATHS ===
# v2/legacy/config.py — anchor everything to _admin/ via parents[3]
# (legacy → v2 → api → _admin)
BASE_DIR = Path(__file__).resolve().parents[3]
# _data: prefer parent (local dev), fallback to inside _admin (Railway)
_data_parent = BASE_DIR.parent / "_data"
_data_local = BASE_DIR / "_data"
DATA_DIR = _data_parent if _data_parent.exists() else _data_local
DATA_DIR.mkdir(parents=True, exist_ok=True)
_proj_parent = BASE_DIR.parent / "projetos"
_proj_local = BASE_DIR / "projetos"
PROJETOS_DIR = _proj_parent if _proj_parent.exists() else _proj_local
PROJETOS_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DB_PATH = BASE_DIR / "projects_db.json"
BANK_TRANSACTION_RULES_PATH = BASE_DIR / "bank_transaction_rules.json"


# === PROJECTS (JSON + PostgreSQL sync) ===

def load_projects() -> dict:
    """Load projects, cached per user_id (read from db_storage ContextVar).

    Bottleneck context: legacy `df.apply(get_project_by_sku)` calls this for
    every vendas row → without cache that's ~2000 DB roundtrips per /reports.
    """
    from .db_storage import _current_user_id
    return _load_projects_cached(_current_user_id())


@functools.lru_cache(maxsize=32)
def _load_projects_cached(user_id: int | None) -> dict:
    return _load_projects_from_db_or_json()


def _load_projects_from_db_or_json() -> dict:
    """Internal: fetch projects from PostgreSQL or JSON (no caching).
    If DATABASE_URL is set (production) — ONLY use DB, never JSON.
    JSON fallback is strictly for local dev without DATABASE_URL.
    """
    import os
    has_db_url = bool(os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL"))

    if has_db_url:
        # Production: per-user data from PostgreSQL only
        try:
            from .db_storage import db_load, db_is_available
            if db_is_available():
                db_data = db_load("projects")
                if db_data and isinstance(db_data, dict):
                    return db_data
        except Exception:
            pass
        return {}  # No projects for this user — never fall back to shared JSON

    # Local dev without DB: use JSON file
    if PROJECTS_DB_PATH.exists():
        with open(PROJECTS_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _invalidate_projects_cache():
    """Clear the per-user projects cache (call after save_projects)."""
    _load_projects_cached.cache_clear()


def save_projects(projects: dict):
    """Save projects to JSON file AND PostgreSQL."""
    _invalidate_projects_cache()
    # JSON (local dev / cache)
    with open(PROJECTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)
    # Create folder if needed
    for proj_id in projects:
        (PROJETOS_DIR / proj_id).mkdir(parents=True, exist_ok=True)
    # PostgreSQL (Railway persistence)
    try:
        from .db_storage import db_save
        db_save("projects", projects)
    except Exception:
        pass


def get_compensation_mode(project: dict) -> str:
    """Вознаграждение: rental | profit_share («фикс» = доля от прибыли, %)."""
    m = project.get("compensation_mode")
    r = project.get("rental")
    has_rental = isinstance(r, dict) and (
        bool(r.get("payments"))
        or (r.get("rate_usd") is not None and r.get("rate_usd") != "")
    )

    if m == "rental":
        return "rental"
    # legacy: fixed = ошибочно как USD; считаем долей от прибыли
    if m in ("profit_share", "fixed"):
        return "profit_share"
    if m == "none" or m is None:
        if has_rental:
            return "rental"
        if project.get("profit_share_pct") is not None:
            try:
                float(project["profit_share_pct"])
                return "profit_share"
            except (TypeError, ValueError):
                pass
        return "profit_share"
    return "rental" if has_rental else "profit_share"


def add_project(
    project_id: str,
    project_type: str,
    description: str,
    sku_prefixes: list[str] | None = None,
    compensation_mode: str = "profit_share",
    profit_share_pct: float | None = None,
    launch_date: str | None = None,  # или date — сериализуется в YYYY-MM-DD
    opening_balance: float | None = None,
    balance_date: str | None = None,
    bank_accounts: list[dict] | None = None,
    ml_account: str | None = None,
    ml_cnpj: str | None = None,
):
    """Новый проект: аренда (rental в JSON) или доля от прибыли % (profit_share)."""
    if compensation_mode not in ("rental", "profit_share"):
        compensation_mode = "profit_share"
    pct_val = None
    if compensation_mode == "profit_share" and profit_share_pct is not None:
        try:
            pct_val = max(0.0, min(100.0, float(profit_share_pct)))
        except (TypeError, ValueError):
            pct_val = None
    projects = load_projects()
    ld = None
    if launch_date is not None and launch_date != "":
        if hasattr(launch_date, "isoformat"):
            ld = launch_date.isoformat()[:10]
        else:
            s = str(launch_date).strip()[:10]
            if s:
                ld = s
    bd = None
    if balance_date is not None and balance_date != "":
        if hasattr(balance_date, "isoformat"):
            bd = balance_date.isoformat()[:10]
        else:
            s = str(balance_date).strip()[:10]
            if s:
                bd = s
    projects[project_id.upper()] = {
        "type": project_type,
        "description": description,
        "status": "pending",
        "sku_prefixes": sku_prefixes or [],
        "mlb_fallback": [],
        "compensation_mode": compensation_mode,
        "profit_share_pct": pct_val,
        "launch_date": ld,
        "rental": None,
        "opening_balance": opening_balance,
        "balance_date": bd,
        "bank_accounts": bank_accounts or [],
        "ml_account": ml_account,
        "ml_cnpj": ml_cnpj,
    }
    save_projects(projects)
    (PROJETOS_DIR / project_id.upper()).mkdir(parents=True, exist_ok=True)


def delete_project(project_id: str):
    """Delete a project (keeps data files)."""
    projects = load_projects()
    if project_id in projects:
        del projects[project_id]
        save_projects(projects)


# Поля проекта, которые можно менять из UI (остальное не трогаем)
_PROJECT_EDITABLE_KEYS = frozenset({
    "type", "description", "status", "sku_prefixes",
    "report_period", "last_report", "next_close",
    "compensation_mode", "profit_share_pct", "mlb_fallback",
    "launch_date", "opening_balance", "balance_date", "bank_accounts",
    "ml_account", "ml_cnpj",
    "company_name", "company_cnpj", "company_state",
    "manual_publicidade", "ff_manual",
    "storage_mode", "aluguel_mensal",
    "tax_regime", "simples_anexo",  # Brazilian tax regime + Simples anexo
    "billing_cycle_day",             # int 1..28 — день закрытия цикла Mercado Ads
    "publicidade_csv_window",        # {from: ISO, to: ISO} | None — сужение CSV-окна
    "publicidade_fill_avg",          # bool — усреднить непокрытые дни рекламы
    "armazenagem_fill_avg",          # bool — усреднить непокрытые дни хранения
    # Balance sheet inputs (ручной ввод через UI CapitalObligationsPanel):
    "initial_equity_brl",            # float — стартовые вложения собственника
    "das_override_pct",              # float 0..100 | None — ручной процент DAS
    "ml_only_revenue",               # bool — продаёт только на ML (разрешает прогрессивный DAS по RBT12)
    # Per-project monthly fixed costs (для break-even tracker per-sale TG):
    # {armazenagem, aluguel, salaries, utilities, software, outros} → float >= 0
    "fixed_costs_monthly",
})

# Канонические категории fixed_costs_monthly. Любые другие keys в input
# отбрасываются. Negative values clamped to 0.
#
# ВАЖНО: armazenagem и aluguel НЕ ВКЛЮЧЕНЫ — они уже считаются автоматически
# из uploaded reports / aluguel_mensal field проекта. Если бы мы их сюда
# включили — было бы дублирование, и пользователь регулярно бы забывал
# обновлять config когда ML не списал хранение или изменилась аренда.
# Здесь только manual-input категории, которые backend не парсит автоматом.
FIXED_COST_CATEGORIES = (
    "salaries", "utilities", "software", "outros",
)

_VALID_TAX_REGIMES = frozenset({"", "simples_nacional", "lucro_presumido"})
_VALID_SIMPLES_ANEXOS = frozenset({"", "I", "II", "III"})


def update_project(
    project_id: str,
    fields: dict,
    rental_fields: dict | None = None,
) -> bool:
    """Обновить разрешённые поля проекта; rental_fields мержится в существующий rental."""
    projects = load_projects()
    pid = project_id.upper()
    if pid not in projects:
        return False
    p = projects[pid]

    for key, val in fields.items():
        if key not in _PROJECT_EDITABLE_KEYS:
            continue
        if key in ("report_period", "last_report", "next_close", "launch_date") and val == "":
            p[key] = None
            continue
        if key in ("last_report", "next_close") and val is not None and hasattr(val, "isoformat"):
            p[key] = val.isoformat()[:10]
            continue
        if key == "launch_date":
            if val is None:
                p[key] = None
            elif hasattr(val, "isoformat"):
                p[key] = val.isoformat()[:10]
            else:
                s = str(val).strip()[:10]
                p[key] = s if s else None
            continue
        if key == "profit_share_pct":
            if val is None:
                p[key] = None
            else:
                try:
                    p[key] = max(0.0, min(100.0, float(val)))
                except (TypeError, ValueError):
                    p[key] = None
            continue
        if key == "compensation_mode":
            if val == "fixed":
                val = "profit_share"
            if val not in ("none", "rental", "profit_share"):
                continue
        if key == "sku_prefixes" and isinstance(val, str):
            p[key] = [s.strip() for s in val.split(",") if s.strip()]
            continue
        if key == "mlb_fallback" and isinstance(val, str):
            p[key] = [s.strip() for s in val.replace("\n", ",").split(",") if s.strip()]
            continue
        if key == "tax_regime":
            s = str(val or "").strip().lower()
            if s not in _VALID_TAX_REGIMES:
                continue
            p[key] = s or None
            continue
        if key == "simples_anexo":
            s = str(val or "").strip().upper()
            if s not in _VALID_SIMPLES_ANEXOS:
                continue
            p[key] = s or None
            continue
        if key == "billing_cycle_day":
            if val is None or val == "":
                p[key] = None
            else:
                try:
                    d = int(val)
                    p[key] = d if 1 <= d <= 28 else None
                except (TypeError, ValueError):
                    p[key] = None
            continue
        if key == "publicidade_csv_window":
            # None | {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}
            if val is None or val == "":
                p[key] = None
            elif isinstance(val, dict):
                f = str(val.get("from") or "").strip()[:10]
                t = str(val.get("to") or "").strip()[:10]
                if f and t:
                    p[key] = {"from": f, "to": t}
                else:
                    p[key] = None
            else:
                p[key] = None
            continue
        if key in ("publicidade_fill_avg", "armazenagem_fill_avg"):
            p[key] = bool(val)
            continue
        if key == "fixed_costs_monthly":
            # Per-project monthly fixed costs. Канонизируем: keep only known
            # categories (FIXED_COST_CATEGORIES), coerce to float >= 0.
            # Sum_monthly считается на лету в API (не сохраняем — производное).
            if val is None or val == "":
                p[key] = None
                continue
            if not isinstance(val, dict):
                continue
            cleaned: dict[str, float] = {}
            for cat in FIXED_COST_CATEGORIES:
                raw = val.get(cat)
                try:
                    f = max(0.0, float(raw)) if raw is not None and raw != "" else 0.0
                except (TypeError, ValueError):
                    f = 0.0
                cleaned[cat] = round(f, 2)
            p[key] = cleaned
            continue
        p[key] = val

    # Invariant: simples_anexo имеет смысл только при simples_nacional.
    if p.get("tax_regime") != "simples_nacional":
        if p.get("simples_anexo") is not None:
            p["simples_anexo"] = None

    if rental_fields:
        r = p.get("rental")
        if r is None or not isinstance(r, dict):
            r = {
                "payments": [],
                "total_paid_usd": 0,
                "total_pending_usd": 0,
                "due_dates": [],
            }
        else:
            r = dict(r)
        if "rate_usd" in rental_fields and rental_fields["rate_usd"] is not None:
            try:
                r["rate_usd"] = float(rental_fields["rate_usd"])
            except (TypeError, ValueError):
                pass
        if rental_fields.get("period") in ("month", "quarter"):
            r["period"] = rental_fields["period"]
        if "note" in rental_fields and rental_fields["note"] is not None:
            r["note"] = str(rental_fields["note"])
        if "next_payment_date" in rental_fields:
            npd = rental_fields["next_payment_date"]
            if npd is None or npd == "":
                r["next_payment_date"] = None
            elif hasattr(npd, "isoformat"):
                r["next_payment_date"] = npd.isoformat()[:10]
            else:
                s = str(npd).strip()[:10]
                r["next_payment_date"] = s if s else None
        p["rental"] = r

    if p.get("compensation_mode") == "fixed":
        p["compensation_mode"] = "profit_share"
    # Устаревший «фикс USD» в JSON — не используем
    if p.get("compensation_mode") == "profit_share":
        p["fixed_fee_usd"] = None
        p["fixed_period"] = None
    elif p.get("compensation_mode") == "rental":
        p["fixed_fee_usd"] = None
        p["fixed_period"] = None

    save_projects(projects)
    return True


PROJECTS = load_projects()

def mlb_url(mlb_code: str) -> str:
    """Build ML product URL from MLB code. Inserts dash: MLB1234 → MLB-1234."""
    mlb = str(mlb_code).strip()
    if not mlb or mlb.lower() == "nan":
        return ""
    # Insert dash after MLB prefix if missing: MLB1234567890 → MLB-1234567890
    import re
    if re.match(r"^MLB\d", mlb):
        mlb = "MLB-" + mlb[3:]
    elif re.match(r"^MLM\d", mlb):  # Mexico
        mlb = "MLM-" + mlb[3:]
    return f"https://produto.mercadolivre.com.br/{mlb}"


# === SKU → PROJECT MAPPING ===
# Built dynamically in get_project_by_sku() to avoid stale module-level state

# Legacy module-level vars (used by app.py for display)
SKU_PREFIXES = {pid: p.get("sku_prefixes", []) for pid, p in PROJECTS.items()} if PROJECTS else {}
ALL_MLB_FALLBACK = {}
for _pid, _p in PROJECTS.items():
    for _mlb in _p.get("mlb_fallback", []):
        ALL_MLB_FALLBACK[_mlb] = _pid
ARTUR_MLBS_FALLBACK = list(ALL_MLB_FALLBACK.keys())


def _build_catalog_project_index() -> dict:
    """Build SKU→project index, cached per user_id."""
    from .db_storage import _current_user_id
    return _build_catalog_project_index_cached(_current_user_id())


@functools.lru_cache(maxsize=32)
def _build_catalog_project_index_cached(user_id: int | None) -> dict:
    idx: dict = {}
    try:
        from .sku_catalog import load_catalog, normalize_sku
        for it in load_catalog():
            nk = normalize_sku(str(it.get("sku", "")))
            proj = str(it.get("project", "")).strip()
            if nk and proj and proj != "NAO_CLASSIFICADO":
                idx[nk] = proj
    except Exception:
        pass
    return idx


def invalidate_catalog_project_index():
    """Clear the per-user catalog index (call after save_catalog)."""
    _build_catalog_project_index_cached.cache_clear()


def _build_mlb_to_sku_from_ml_items() -> dict:
    """MLB → seller_sku map (per-user) из кеша ml_user_items.
    Используется как fallback в get_project_by_sku когда vendas-строка
    пришла без SKU но с MLB. Источники seller_sku в raw JSONB:
    seller_custom_field, attributes[].SELLER_SKU,
    variations[].attributes[].SELLER_SKU,
    variations[].attribute_combinations[].SELLER_SKU.
    """
    from .db_storage import _current_user_id
    return _build_mlb_to_sku_from_ml_items_cached(_current_user_id())


@functools.lru_cache(maxsize=32)
def _build_mlb_to_sku_from_ml_items_cached(user_id: int | None) -> dict:
    import json as _json
    idx: dict = {}
    if user_id is None:
        return idx
    from .db_storage import _get_db, _put_db
    conn = _get_db()
    if conn is None:
        return idx
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT item_id, raw FROM ml_user_items WHERE user_id = %s",
            (user_id,),
        )
        for item_id, raw in cur.fetchall():
            iid = (item_id or "").strip()
            if not iid:
                continue
            if isinstance(raw, str):
                try:
                    raw = _json.loads(raw)
                except Exception:
                    raw = {}
            if not isinstance(raw, dict):
                raw = {}

            picked_sku = ""

            def _take(val):
                nonlocal picked_sku
                if picked_sku:
                    return
                s = str(val or "").strip()
                if s and s.lower() != "nan":
                    picked_sku = s

            if raw.get("seller_custom_field"):
                _take(raw.get("seller_custom_field"))
            for attr in raw.get("attributes") or []:
                if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                    _take(attr.get("value_name") or attr.get("value_id"))
            # Для вариаций берём SKU первой вариации (для родительского MLB
            # любая вариация ведёт на тот же листинг → проект один)
            for var in raw.get("variations") or []:
                if not isinstance(var, dict):
                    continue
                for attr in var.get("attributes") or []:
                    if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                        _take(attr.get("value_name") or attr.get("value_id"))
                for attr in var.get("attribute_combinations") or []:
                    if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                        _take(attr.get("value_name") or attr.get("value_id"))
                if picked_sku:
                    break

            if picked_sku:
                idx[iid] = picked_sku
    except Exception:
        pass
    finally:
        _put_db(conn)
    return idx


def invalidate_mlb_to_sku_from_ml_items():
    """Сброс per-user MLB→SKU индекса (звать после refresh ml_user_items)."""
    _build_mlb_to_sku_from_ml_items_cached.cache_clear()


def get_project_by_sku(sku: str, mlb: str = "") -> str:
    """Determine project from SKU or MLB code.
    Priority:
      1) catalog direct mapping (sku → project)
      2) sku_prefixes
      3) mlb_fallback (явный список MLB на проекте)
      4) ml_user_items lookup: MLB → seller_sku → catalog/prefix
         (фолбэк для строк vendas с пустым SKU но с MLB)
      5) single project (если один проект всего)
    """
    sku = sku.strip()
    mlb = mlb.strip()
    sku_upper = sku.upper()

    # 1) Direct mapping from catalog index (cached, single DB call)
    cat_idx = _build_catalog_project_index()
    if sku_upper in cat_idx:
        return cat_idx[sku_upper]

    # 2) Prefix-based matching + 3) mlb_fallback
    projects = load_projects()
    for pid, p in projects.items():
        for prefix in p.get("sku_prefixes", []):
            if prefix and sku.lower().startswith(prefix.lower()):
                return pid
        for fb_mlb in p.get("mlb_fallback", []):
            if fb_mlb and mlb == fb_mlb:
                return pid

    # 4) ML cache fallback: SKU пустой/неизвестный, но MLB есть → look up seller_sku
    if mlb and not sku_upper:
        mlb_idx = _build_mlb_to_sku_from_ml_items()
        resolved_sku = mlb_idx.get(mlb, "")
        if resolved_sku:
            resolved_upper = resolved_sku.strip().upper()
            if resolved_upper in cat_idx:
                return cat_idx[resolved_upper]
            for pid, p in projects.items():
                for prefix in p.get("sku_prefixes", []):
                    if prefix and resolved_sku.lower().startswith(prefix.lower()):
                        return pid

    # 5) Single project fallback
    if len(projects) == 1:
        return list(projects.keys())[0]
    return "NAO_CLASSIFICADO"


# === DATA SOURCES ===
# Each source: id, display name, file pattern, frequency, applies_to
DATA_SOURCES = {
    # --- ECOM sources (shared across ecom projects) ---
    "vendas_ml": {
        "name": "Vendas ML (Relatorio de vendas)",
        "description": "Relatorio completo de vendas Mercado Livre com SKU, MLB, valores bruto/NET",
        "file_pattern": "vendas_ml*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "ML > Vendas > Relatorio de vendas > Exportar CSV",
        "columns_key": "# de anuncio",  # to auto-detect
    },
    "collection_mp": {
        "name": "Collection MP (Cobrancas)",
        "description": "Relatorio de cobrancas do Mercado Pago — net_received_amount real",
        "file_pattern": "collection_mp*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "MP > Relatorios > Cobrancas > Exportar CSV",
        "columns_key": "net_received_amount",
    },
    "extrato_mp": {
        "name": "Extrato Mercado Pago",
        "description": "Extrato de conta MP — todas movimentacoes (entradas/saidas)",
        "file_pattern": "extrato_mp*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "MP > Extrato > Exportar CSV",
        "columns_key": "description",
    },
    "fatura_ml": {
        "name": "Fatura ML (Faturamento)",
        "description": "Fatura mensal ML — comissoes, tarifas envio, armazenagem",
        "file_pattern": "fatura_ml*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "ML > Faturamento > Detalhes da fatura > Exportar",
        "columns_key": "Tarifa de venda",
    },
    "ads_publicidade": {
        "name": "Anuncios Patrocinados (Ads)",
        "description": "Relatorio de ads — investimento, ACOS, ROAS por anuncio",
        "file_pattern": "ads_publicidade*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "ML > Publicidade > Relatorios > Exportar CSV",
        "columns_key": "Investimento",
    },
    "armazenagem_full": {
        "name": "Custos Armazenagem Full",
        "description": "Custos de armazenamento no Full por periodo",
        "file_pattern": "armazenagem_full*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "ML > Full > Custos por servico de armazenamento",
        "columns_key": "Tarifa por unidade",
    },
    "retirada_full": {
        "name": "Custos Retirada Full (вывоз/утилизация)",
        "description": "Custo por retirada de estoque Full — Envio para o endereço (вывоз) e Descarte (утилизация). Tarifa por unidade + base para COGS списание.",
        "file_pattern": "Relatorio_Tarifas_Full*.xlsx",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "ML > Full > Tarifas Full > Custo por retirada de estoque",
        "columns_key": "Forma de retirada",
    },
    "stock_full": {
        "name": "Stock ML Full (по SKU)",
        "description": "Relatório geral de estoque ML Full — qtd unidades por SKU",
        "file_pattern": "stock_full*.xlsx",
        "frequency": "weekly",
        "type": "ecom",
        "export_from": "ML > Full > Estoque > Exportar relatório geral",
        "columns_key": None,
    },
    "dados_fiscais": {
        "name": "Dados Fiscais ML (catálogo)",
        "description": "Catálogo oficial ML — SKU × MLB × NCM × Origem × Custo × Peso × CSOSN. Fonte autoritativa para себестоимость и налоги на уровне SKU.",
        "file_pattern": "Dados_Fiscais*.xlsx",
        "frequency": "on_demand",
        "type": "ecom",
        "export_from": "ML > Minha conta > Dados fiscais > Exportar Excel",
        "columns_key": "Código do Anúncio",
    },
    "full_express": {
        "name": "Fatura Full Express (3PL)",
        "description": "Fatura de preparo, seguro, frete — 3PL",
        "file_pattern": "full_express*.{csv,pdf}",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "Solicitar ao operador 3PL",
        "columns_key": None,
    },
    "after_collection": {
        "name": "Pos-vendas (Devolucoes/Reclamacoes)",
        "description": "Relatorio pos-venda — devolucoes, reembolsos, reclamacoes",
        "file_pattern": "after_collection*.csv",
        "frequency": "monthly",
        "type": "ecom",
        "export_from": "MP > Relatorios > Pos-vendas > Exportar",
        "columns_key": "amount_refunded",
    },
    # --- BANK sources (shared) ---
    "extrato_nubank": {
        "name": "Extrato Nubank PJ",
        "description": "Extrato bancario Nubank — compras, transferencias, DAS",
        "file_pattern": "extrato_nubank*.csv",
        "frequency": "monthly",
        "type": "all",
        "export_from": "Nubank > Extrato > Exportar CSV",
        "columns_key": "Identificador",
    },
    "fatura_nubank": {
        "name": "Fatura Nubank Crédito",
        "description": "Fatura do cartão de crédito Nubank PJ — compras parceladas e à vista",
        "file_pattern": "fatura_nubank*.csv",
        "frequency": "monthly",
        "type": "all",
        "export_from": "Nubank > Cartão > Fatura > Exportar CSV",
        "columns_key": None,
    },
    # --- SERVICES sources (Estonia/Ganza) ---
    "extrato_c6_brl": {
        "name": "Extrato C6 (BRL)",
        "description": "Extrato C6 conta BRL — recebe de Nubank, compra USD",
        "file_pattern": "extrato_c6_brl*.{csv,pdf,xlsx}",
        "frequency": "monthly",
        "type": "services",
        "export_from": "C6 > Conta Global BRL > Extrato > Exportar",
        "columns_key": None,
    },
    "extrato_c6_usd": {
        "name": "Extrato C6 (USD)",
        "description": "Extrato C6 conta USD — saldo USD, pagamentos TrafficStars",
        "file_pattern": "extrato_c6_usd*.{csv,pdf,xlsx}",
        "frequency": "monthly",
        "type": "services",
        "export_from": "C6 > Conta Global USD > Extrato > Exportar",
        "columns_key": None,
    },
    "trafficstars": {
        "name": "TrafficStars Report",
        "description": "Relatorio de gastos TrafficStars — campanhas GANZA (USD)",
        "file_pattern": "trafficstars*.csv",
        "frequency": "monthly",
        "type": "services",
        "export_from": "TrafficStars > Reports > Export",
        "columns_key": None,
    },
    "invoices_estonia": {
        "name": "Invoices Estonia (SHPS)",
        "description": "Notas fiscais/invoices da Estonia — entradas + imposto",
        "file_pattern": "invoices_estonia*.csv",
        "frequency": "on_demand",
        "type": "services",
        "export_from": "Contador / manual",
        "columns_key": None,
    },
    "das_simples": {
        "name": "DAS Simples Nacional",
        "description": "Documento de Arrecadação do Simples Nacional (PDF do contador)",
        "file_pattern": "das_simples*.pdf",
        "frequency": "monthly",
        "type": "all",
        "export_from": "Contadora envia mensalmente (PGDASD-DAS-*.pdf)",
        "columns_key": None,
    },
    "nfse_shps": {
        "name": "NFS-e (Nota Fiscal Estonia)",
        "description": "Nota Fiscal de Serviço eletrônica emitida para SHPS (Estonia)",
        "file_pattern": "nfse_shps*.pdf",
        "frequency": "monthly",
        "type": "services",
        "export_from": "Portal NFS-e Campos do Jordão",
        "columns_key": None,
    },
    "bybit_history": {
        "name": "Bybit P2P History",
        "description": "Historico de transacoes P2P Bybit — USDT envios/recebimentos",
        "file_pattern": "bybit_history*.csv",
        "frequency": "on_demand",
        "type": "services",
        "export_from": "Bybit > P2P > Order History > Export",
        "columns_key": None,
    },
}

# === KNOWN PASSWORDS (for protected bank statements) ===
KNOWN_PASSWORDS = [
    "716816",   # C6 Bank
    "456595",   # C6 Bank
]

# === ESTONIA / GANZA — Cash Flow Logic ===
#
# TWO possible flows (transition date TBD):
#
# CURRENT FLOW:
#   Invoice SHPS → Nubank PJ (BRL)
#     → Transfer Nubank → C6 PJ (BRL)
#       → Buy USD inside C6 (BRL→USD, exchange rate)
#         → Pay TrafficStars (USD)
#
# FUTURE FLOW (requested, date unknown):
#   Invoice SHPS → C6 PJ (BRL) directly
#     → Buy USD inside C6
#       → Pay TrafficStars (USD)
#
# The system must handle BOTH — detect by checking which account
# received the SHPS payment (Nubank or C6).
#
# Key accounts:
#   Nubank PJ: receives invoices (current), pays suppliers, transfers to C6
#   C6 PJ (BRL): receives from Nubank (current) or SHPS (future), converts to USD
#   C6 PJ (USD): holds USD, pays TrafficStars
#
# TAX: Estonian tax rate depends on REVENUE LEVEL (not period).
# Rates: to be confirmed by user. Known values from historical data:
#   - 15.50% (used Jul-Sep 2025)
#   - 16.75% (used Oct 2025 onwards)
# TODO: get exact revenue thresholds for rate changes
# Trade (товары) DAS rate — for ML/Empresa products
# Simples Nacional Anexo I bracket
TRADE_DAS_RATE = 0.045  # 4.5%

# Progressive tax brackets (imposto servico):
#   Revenue up to R$ 180.000 → 15,50%
#   R$ 180.000 — R$ 360.000 → 16,75%
#   R$ 360.000 — R$ 720.000 → 18,75%
#   R$ 720.000 — R$ 1.800.000 → 19,75% (next bracket)
# Commission charged to client = same % as tax bracket
ESTONIA_TAX_BRACKETS = [
    {"limit": 180000,  "rate": 0.155},    # 15,50%
    {"limit": 360000,  "rate": 0.1675},   # 16,75%
    {"limit": 720000,  "rate": 0.1875},   # 18,75%
    {"limit": 1800000, "rate": 0.1975},   # 19,75%
]

# Historical rates actually applied per invoice line
# (some invoices span two brackets)
ESTONIA_TAX_APPLIED = {
    "2025-07-07": 0.155,
    "2025-07-09": 0.155,
    "2025-08-08": 0.155,
    "2025-09-03": 0.155,
    "2025-11-05": [0.155, 0.1675],  # split across brackets
    "2025-10-02": 0.1675,
    "2025-12-02": 0.1675,
    "2026-01-14": [0.1675, 0.1875],  # split across brackets
    "2026-02-02": 0.1875,
    "2026-03-03": 0.1875,
}

# C6 Bank — two sub-accounts
C6_ACCOUNTS = {
    "brl": "Conta Global BRL",
    "usd": "Conta Global USD",
}

# === TRANSACTION CLASSIFICATION RULES ===
# Редактируемая копия: bank_transaction_rules.json (страница «Правила банковских выписок»).
# Each rule: keywords (in description, lowercase) → category, project, label
# Categories: internal_transfer, expense, income, tax, personal, supplier, ads, fulfillment
DEFAULT_TRANSACTION_RULES = [
    # Internal transfers
    {"keywords": ["ganza comercial", "bco c6", "c6 s.a"], "category": "internal_transfer", "project": "GANZA", "label": "Перевод BRL → C6 (Ganza)"},
    # C6 internal: BRL → USD conta global
    {"keywords": ["transferência para c6 conta global", "transferencia para c6 conta global", "c6 conta global", "câmbio"], "category": "fx", "project": "GANZA", "label": "💱 Câmbio C6 (BRL→USD)"},
    # C6 card fees
    {"keywords": ["emissão cartão global", "emissao cartao global", "emiss. cartao global"], "category": "bank_fee", "project": "GANZA", "label": "💳 Эмиссия C6 Global Card"},
    # Generic Pix outgoing
    {"keywords": ["transf enviada pix", "pix enviado para sabesp"], "category": "expense", "project": None, "label": "💸 PIX исходящий"},

    # Mercado Pago / Shopee
    {"keywords": ["shpp brasil", "shpp"], "category": "income", "project": "GANZA", "label": "Поступление от Mercado Pago/Shopee"},

    # Mercado Pago TRANSACTION_TYPE values
    # Dinheiro retido — money held by ML (separate group)
    {"keywords": ["dinheiro retido"], "category": "refund", "project": None, "label": "🔒 Retido ML"},
    # Devoluções/Reclamações — actual returns/refunds (separate group)
    {"keywords": ["devoluções e reclamações", "devolucoes e reclamacoes",
                  "débito por dívida devoluções", "debito por divida devolucoes",
                  "débito por dívida reclamações", "debito por divida reclamacoes",
                  "dívida reclamaç", "dívida devolu"],
     "category": "refund", "project": None, "label": "↩️ Devoluções/Reclamações ML"},
    # Fatura ML — реклама + хранение Full + комиссии
    {"keywords": ["faturas vencidas do mercado livre", "fatura mercado livre"], "category": "expense", "project": None, "label": "📋 Fatura ML (реклама + Full)"},
    # Generic "Débito por dívida" — fallback (catches anything else)
    {"keywords": ["débito por dívida", "debito por divida"], "category": "expense", "project": None, "label": "📋 Fatura ML (débito por dívida)"},
    {"keywords": ["liberação de dinheiro", "liberacao de dinheiro"], "category": "income", "project": None, "label": "💰 Liberação venda ML"},
    {"keywords": ["pagamento de venda", "venda direta"], "category": "income", "project": None, "label": "💰 Venda ML"},
    {"keywords": ["recebimento por marketing"], "category": "ads", "project": None, "label": "📢 Reembolso publicidade"},
    {"keywords": ["rendimento", "remuneração"], "category": "income", "project": "GANZA", "label": "💵 Rendimento MP"},
    {"keywords": ["pagamento de tarifa de envio", "tarifa de envio"], "category": "shipping", "project": None, "label": "📮 Tarifa envio ML"},
    {"keywords": ["devolução", "devolucao", "estorno de venda"], "category": "refund", "project": None, "label": "↩️ Devolução cliente"},
    {"keywords": ["transferência para conta bancária", "saque"], "category": "internal_transfer", "project": "GANZA", "label": "Saque MP → banco"},
    {"keywords": ["pagamento de mercadoria"], "category": "supplier", "project": None, "label": "📦 Compra mercadoria"},

    # Suppliers - bags (Artur)
    {"keywords": ["joed comercio", "joed eletro"], "category": "supplier", "project": "ARTUR", "label": "Закупка JOED (сумки Артура)"},
    {"keywords": ["mariz"], "category": "supplier", "project": "ARTUR", "label": "Закупка Mariz (сумки)"},
    {"keywords": ["conquista comercio", "conquista utilidades"], "category": "supplier", "project": "ARTUR", "label": "Закупка Conquista (сумки)"},

    # Full Express / fulfillment — common to ALL ecom projects, requires manual split
    {"keywords": ["full express"], "category": "fulfillment", "project": None, "label": "Full Express (определить проект ⚠️)"},
    {"keywords": ["leticia"], "category": "fulfillment", "project": None, "label": "3PL (определить проект ⚠️)"},
    {"keywords": ["box-70", "box 70"], "category": "fulfillment", "project": None, "label": "Box-70 3PL (определить проект ⚠️)"},

    # International / TrafficStars
    {"keywords": ["topazio", "ebanx"], "category": "expense", "project": "GANZA", "label": "Международный платёж (Topazio/EBANX)"},
    {"keywords": ["caliza"], "category": "expense", "project": "ESTONIA", "label": "CALIZA (AdvertMedia)"},
    {"keywords": ["trafficstars", "traffic stars"], "category": "expense", "project": "GANZA", "label": "TrafficStars"},

    # Accounting / professional services
    {"keywords": ["swp", "contab"], "category": "accounting", "project": "GANZA", "label": "🧾 SWP Contabilidade (бухгалтер)"},

    # Shipping / Correios
    {"keywords": ["correios", "telegrafos", "ect "], "category": "shipping", "project": None, "label": "📮 Correios (доставка)"},

    # Utilities (light, water, internet)
    {"keywords": ["energia", "elektro", "eletro paulista", "edp ", "sabesp", "vivo ", "claro ", "tim ", "oi "], "category": "utilities", "project": "GANZA", "label": "💡 Коммунальные услуги"},

    # Bank fees
    {"keywords": ["tarifa", "seguro conta", "anuidade"], "category": "bank_fee", "project": "GANZA", "label": "🏦 Банковская комиссия"},

    # Software / SaaS
    {"keywords": ["aws", "google", "microsoft", "github", "openai", "stripe"], "category": "software", "project": "GANZA", "label": "💻 SaaS подписка"},

    # Refunds / chargebacks
    {"keywords": ["reembolso", "estorno cliente"], "category": "refund", "project": "GANZA", "label": "↩️ Возврат клиенту"},

    # Boleto/document fees
    {"keywords": ["certificacao", "certificado"], "category": "expense", "project": "GANZA", "label": "📄 Сертификация"},

    # Travel / fuel
    {"keywords": ["o j santos", "turismo", "uber", "99 pop"], "category": "expense", "project": "GANZA", "label": "🚗 Транспорт/поездки"},

    # Tax / DAS
    {"keywords": ["das", "simples nacional", "receita federal"], "category": "tax", "project": "GANZA", "label": "DAS Simples Nacional"},
    {"keywords": ["sefaz", "dare"], "category": "tax", "project": "GANZA", "label": "Налог штата"},
    {"keywords": ["pref mun", "prefeitura"], "category": "tax", "project": "GANZA", "label": "Муниципальный налог"},

    # Personal — кафе, рестораны, мини-маркеты в Campos do Jordão
    {"keywords": ["mini mercado", "minimercado", "equilbrio", "equilibrio"], "category": "personal", "project": "GANZA", "label": "🛒 Мини-маркет (личное)"},
    {"keywords": ["um pe de cafe", "café ", "padaria", "restaurante", "lanchonete", "tgr "], "category": "personal", "project": "GANZA", "label": "☕ Кафе/Ресторан (личное)"},
    {"keywords": ["dlknet", "loja do tadeu", "santos magalhaes", "beiraflor", "serra da estrela"], "category": "personal", "project": "GANZA", "label": "🛒 Местный магазин (личное)"},
    {"keywords": ["cvs"], "category": "personal", "project": "GANZA", "label": "💊 Аптека CVS (личное)"},

    # Dividends to Nikolay (owner)
    {"keywords": ["nikolai rapaev", "nikolay rapaev"], "category": "dividends", "project": "GANZA", "label": "💎 Дивиденды Николая"},

    # Refunds / estorno
    {"keywords": ["estorno", "reembolso", "mdr solucao"], "category": "income", "project": "GANZA", "label": "Возврат / эсторно"},

    # Credit card
    {"keywords": ["pagamento de fatura"], "category": "expense", "project": "GANZA", "label": "Оплата кредитки"},
    {"keywords": ["pagamento recebido"], "category": "income", "project": "GANZA", "label": "Поступление"},

    # SHPS Estonia (incoming invoices)
    {"keywords": ["shps", "santander"], "category": "income", "project": "ESTONIA", "label": "Инвойс от Estonia (SHPS)"},
]

_rules_cache: list | None = None
_rules_cache_mtime: float | None = None


def invalidate_transaction_rules_cache() -> None:
    global _rules_cache, _rules_cache_mtime
    _rules_cache = None
    _rules_cache_mtime = None


def _normalize_rule_dict(r: dict) -> dict | None:
    if not isinstance(r, dict):
        return None
    kws = r.get("keywords")
    if isinstance(kws, str):
        kws = [x.strip() for x in kws.replace("\n", ",").split(",") if x.strip()]
    elif isinstance(kws, list):
        kws = [str(x).strip() for x in kws if str(x).strip()]
    else:
        kws = []
    if not kws:
        return None
    cat = str(r.get("category", "uncategorized")).strip() or "uncategorized"
    label = str(r.get("label", "")).strip() or "?"
    proj = r.get("project")
    if proj in (None, "", "—", "null", "NULL"):
        proj = None
    else:
        proj = str(proj).strip() or None
    return {"keywords": kws, "category": cat, "project": proj, "label": label}


def normalize_transaction_rules_list(raw: list) -> list:
    """Привести список правил к каноническому виду (для сохранения и после импорта)."""
    out = []
    for item in raw:
        n = _normalize_rule_dict(item if isinstance(item, dict) else {})
        if n:
            out.append(n)
    return out


def load_transaction_rules() -> list:
    """Правила: PostgreSQL → JSON файл → DEFAULT. Порядок строк = приоритет."""
    global _rules_cache, _rules_cache_mtime
    path = BANK_TRANSACTION_RULES_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _rules_cache is not None and _rules_cache_mtime == mtime:
        return _rules_cache

    rules: list | None = None

    # Try PostgreSQL first (per-user)
    try:
        from .db_storage import db_load, db_is_available
        if db_is_available():
            db_data = db_load("transaction_rules")
            if db_data and isinstance(db_data, list) and len(db_data) > 0:
                rules = normalize_transaction_rules_list(db_data)
            else:
                # New user — start with defaults, don't read shared JSON
                rules = copy.deepcopy(DEFAULT_TRANSACTION_RULES)
    except Exception:
        pass

    # No DB → local dev, fallback to JSON file
    if not rules and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rules = normalize_transaction_rules_list(data)
            elif isinstance(data, dict) and isinstance(data.get("rules"), list):
                rules = normalize_transaction_rules_list(data["rules"])
        except (json.JSONDecodeError, OSError, TypeError):
            rules = None

    if not rules:
        rules = copy.deepcopy(DEFAULT_TRANSACTION_RULES)

    _rules_cache = rules
    _rules_cache_mtime = mtime
    return _rules_cache


def save_transaction_rules(rules: list) -> None:
    normalized = normalize_transaction_rules_list(rules)
    if not normalized:
        normalized = copy.deepcopy(DEFAULT_TRANSACTION_RULES)
    # JSON (local)
    path = BANK_TRANSACTION_RULES_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    # PostgreSQL (Railway)
    try:
        from .db_storage import db_save
        db_save("transaction_rules", normalized)
    except Exception:
        pass
    invalidate_transaction_rules_cache()
    load_transaction_rules()


def classify_transaction(description: str, value: float = 0) -> dict:
    """Classify a transaction by description. Returns dict with category/project/label."""
    desc_lower = (description or "").lower()
    for rule in load_transaction_rules():
        if any(kw in desc_lower for kw in rule["keywords"]):
            return {
                "category": rule["category"],
                "project": rule["project"],
                "label": rule["label"],
                "confidence": "auto",
            }
    return {
        "category": "uncategorized",
        "project": None,
        "label": "❓ Не классифицировано",
        "confidence": "none",
    }


def __getattr__(name: str):
    """Старый код: `from config import TRANSACTION_RULES` — отдаём актуальный список из JSON."""
    if name == "TRANSACTION_RULES":
        return load_transaction_rules()
    raise AttributeError(f"module '{__name__}' has no attribute '{name!r}'")


# === MONTHS TO TRACK ===
MONTHS = [
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08",
    "2026-09", "2026-10", "2026-11", "2026-12",
]
