import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Atendimento: ML claim codes turned into what the seller reads."""
from v2.services.escalar_atendimento import claim_view


def test_return_with_ml_coverage():
    v = claim_view({
        "id": 1, "type": "returns", "stage": "claim", "status": "closed",
        "resource_id": 200, "reason_id": "PDD9939", "related_entities": ["return"],
        "order_item": {"id": "7469162840", "title": "Bolsa", "unit_price": 67.94, "quantity": 1},
        "resolution": {"reason": "coverage_decision", "benefited": ["complainant"],
                       "closed_by": "mediator", "applied_coverage": True},
    })
    assert v["kind"] == "return"
    assert v["itemId"] == "MLB7469162840"
    assert v["reason"][0] == "Produto diferente ou com defeito"
    assert v["outcome"][0] == "Mercado Livre cobriu o valor"
    assert v["coverage"] is True and v["sellerWon"] is False


def test_mediation_not_received():
    v = claim_view({"type": "mediations", "stage": "dispute", "status": "opened",
                    "reason_id": "PNR3430", "order_item": {"id": "MLB1"}})
    assert (v["kind"], v["mediation"], v["reason"][0]) == ("claim", True, "Produto não recebido")
    assert v["outcome"] is None
