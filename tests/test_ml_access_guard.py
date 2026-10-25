import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Инвариант: перепривязка к владельцу кабинета и проверка доступа — вместе.

Сегодня партнёр не видит чужих карточек по случайности: эндпоинты escalar
читают `ml_user_items` под `user.id`, а данные лежат под id владельца, так что
партнёру отвечает пустота. Дыра появляется в тот момент, когда кто-то делает
«очевидную» правку — подставляет id владельца, как это сделано в финансах, —
и получает выдачу всего кабинета каждому приглашённому.

Поэтому правило механическое: функция, которая обращается к карточкам или к
токену ML не под `user.id`, обязана спросить `ml_access.resolve_scope`. Тест
не проверяет, что фильтр применён правильно — это дело test_ml_access.py. Он
проверяет, что про фильтр вообще не забыли, и падает на новом эндпоинте, а не
на жалобе партнёра.
"""
import ast
import pathlib

ROUTER = pathlib.Path(__file__).resolve().parent.parent / "v2" / "routers" / "escalar.py"
src = ROUTER.read_text()
tree = ast.parse(src)

# Обращения, через которые утекают карточки кабинета.
SENSITIVE = ("ml_user_items", "get_valid_access_token")
SAFE_ID = "user.id"
SCOPE_CALL = "ml_access.resolve_scope"

offenders: list[str] = []
checked = 0

for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    body = ast.get_source_segment(src, node) or ""
    if not any(marker in body for marker in SENSITIVE):
        continue
    checked += 1
    if SCOPE_CALL in body:
        continue
    # Нет scope — значит каждый id, уходящий в запрос, должен быть своим.
    foreign = [
        ident for ident in ("owner_id", "effective_id", "effective_user_id")
        if ident in body
    ]
    if foreign:
        offenders.append(
            f"{node.name} (строка {node.lineno}): обращается к карточкам под "
            f"{', '.join(foreign)}, но не вызывает {SCOPE_CALL}"
        )

# Эндпоинт, который берёт пользователя из тела запроса, а не из cookie,
# действует от чужого имени по определению. Такие зовут только наши вебхуки,
# и FastAPI должен требовать от них общий секрет — иначе любой достаёт до них
# через публичный /api/v2-proxy/* с любым user_id.
for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    decorators = [ast.get_source_segment(src, d) or "" for d in node.decorator_list]
    if not any(d.startswith("router.") for d in decorators):
        continue
    body = ast.get_source_segment(src, node) or ""
    if "body.user_id" not in body:
        continue
    checked += 1
    if not any("require_internal" in d for d in decorators):
        offenders.append(
            f"{node.name} (строка {node.lineno}): действует от body.user_id, "
            "но не требует require_internal"
        )

print(f"функций, работающих с карточками кабинета: {checked}")
if offenders:
    print("\nНАРУШЕНИЯ ИНВАРИАНТА:")
    for o in offenders:
        print("  -", o)
    sys.exit(1)
print("инвариант соблюдён: перепривязка без проверки доступа отсутствует")

# Обратная проверка: сам слой доступа на месте и ни одна из точек входа не
# обходит его, реализуя атрибуцию самостоятельно.
from v2.services import ml_access  # noqa: E402

assert hasattr(ml_access, "resolve_scope")
assert hasattr(ml_access, "filter_rows")
assert hasattr(ml_access.MlScope, "allows_read")
assert hasattr(ml_access.MlScope, "allows_write")

dup = [
    name for name in ("get_project_by_sku",)
    if name in src
]
if dup:
    print(
        "\nВНИМАНИЕ: escalar.py вызывает "
        + ", ".join(dup)
        + " напрямую — атрибуция должна приходить из ml_access, "
        "иначе доступ и P&L разойдутся"
    )
    sys.exit(1)
print("атрибуция не дублируется в роутере")
