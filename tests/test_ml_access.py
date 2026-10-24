import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""ml_access: who may see and touch which listing inside one ML cabinet.

This test exists because the escalar endpoints had no membership check at
all, and the obvious fix — resolve the caller to the project owner, the way
finance does — would have handed every partner the whole cabinet. The owner
namespace is where the data lives; it is not an access decision. So the two
things are pinned here together: a partner is rebound to the owner's data AND
confined to the SKUs of the projects they were invited to.

The cases that matter are the negative ones. A listing whose SKU resolves to
another partner's project must be invisible, and so must a listing whose SKU
resolves to nothing: unattributed stock belongs to the cabinet owner, never
to whoever happens to be looking. Seven of the 240 live listings are in that
state, which is why it is a rule and not an edge case.
"""
import asyncio
from v2.services import ml_access

# One cabinet, owner id 6. Catalog maps SKU → project; two partners hold one
# project each; one listing carries a SKU nobody claims.
# Attribution is whatever get_project_by_sku says in the owner's namespace —
# catalog, SKU prefixes and mlb_fallback together. Listings it cannot place
# come back as NAO_CLASSIFICADO.
ATTRIBUTION = {
    "MLB1": "ARTHUR",
    "MLB2": "ARTHUR",
    "MLB3": "ARTEM-TENIS",
    "MLB4": "GANZA",
    "MLB5": "NAO_CLASSIFICADO",  # SKU nobody claims
    "MLB6": "NAO_CLASSIFICADO",  # no seller_sku at all
}
OWN_PROJECTS = {6: {"ARTHUR", "ARTEM-TENIS", "GANZA", "JOOM"}}
MEMBERSHIPS = {
    7: [{"project_name": "ARTHUR", "owner_id": 6, "role": "admin", "synced": True}],
    8: [{"project_name": "ARTEM-TENIS", "owner_id": 6, "role": "viewer", "synced": True}],
    9: [],  # a logged-in user with no membership anywhere
}

ml_access._attribute_items = lambda owner_id: dict(ATTRIBUTION)
ml_access._load_own_projects = lambda caller_id: set(OWN_PROJECTS.get(caller_id, set()))
ml_access._load_memberships = lambda pool, caller_id: asyncio.sleep(
    0, result=list(MEMBERSHIPS.get(caller_id, []))
)


class _User:
    def __init__(self, uid): self.id = uid


def scope(uid, project=""):
    return asyncio.get_event_loop().run_until_complete(
        ml_access.resolve_scope(None, _User(uid), project=project)
    )


def readable(sc):
    return {i for i in ATTRIBUTION if sc.allows_read(i)}


fails: list[str] = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
    print(("ok   " if got == want else "FAIL ") + label)


# --- owner sees the whole cabinet, including unattributed stock -------------
own = scope(6)
check("owner is_owner", own.is_owner, True)
check("owner item filter absent", own.item_ids, None)
check("owner reads all 6", readable(own), set(ATTRIBUTION))
check("owner writes orphan", own.allows_write("MLB5"), True)

# --- partner A: only ARTHUR ------------------------------------------------
a = scope(7)
check("A rebound to owner", a.owner_id, 6)
check("A not owner", a.is_owner, False)
check("A projects", set(a.projects), {"ARTHUR"})
check("A reads only own SKUs", readable(a), {"MLB1", "MLB2"})
check("A cannot read B", a.allows_read("MLB3"), False)
check("A cannot read unattributed", a.allows_read("MLB5"), False)
check("A cannot read SKU-less", a.allows_read("MLB6"), False)
check("A writes own (admin)", a.allows_write("MLB1"), True)
check("A cannot write B", a.allows_write("MLB3"), False)

# --- partner B: only ARTEM-TENIS, and viewer cannot write ------------------
b = scope(8)
check("B projects", set(b.projects), {"ARTEM-TENIS"})
check("B reads only own", readable(b), {"MLB3"})
check("B cannot read A", b.allows_read("MLB1"), False)
check("B viewer cannot write own", b.allows_write("MLB3"), False)

# --- the two partners never overlap ---------------------------------------
check("A and B disjoint", readable(a) & readable(b), set())

# --- no membership → nothing ----------------------------------------------
n = scope(9)
check("stranger reads nothing", readable(n), set())
check("stranger writes nothing", n.allows_write("MLB1"), False)

# --- asking for a project you are not in leaks nothing --------------------
a_other = scope(7, project="ARTEM-TENIS")
check("A asking for B's project", readable(a_other), set())
check("A asking for B's project: no projects", set(a_other.projects), set())

# --- owner scoped to one project is confined to it ------------------------
own_one = scope(6, project="ARTHUR")
check("owner scoped to ARTHUR", readable(own_one), {"MLB1", "MLB2"})
check("owner scoped: orphan hidden", own_one.allows_read("MLB5"), False)

# --- filter_rows drops foreign rows instead of erroring -------------------
rows = [{"item_id": i, "title": i} for i in ATTRIBUTION]
check(
    "filter_rows for A",
    {r["item_id"] for r in ml_access.filter_rows(a, rows)},
    {"MLB1", "MLB2"},
)
check("filter_rows for owner", len(ml_access.filter_rows(own, rows)), 6)

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all ml_access isolation checks passed")
