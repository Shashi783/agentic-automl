### FILE: service/tools/select.py
from __future__ import annotations
from typing import Dict, Tuple
from contracts.schemas import SelectRequest, SelectResponse, TrainResponse


def _score_tuple(metrics: Dict[str, float], primary: str) -> Tuple[float, float, float]:
    sil = metrics.get("silhouette", float("nan"))
    db = metrics.get("db", float("nan"))
    ch = metrics.get("ch", float("nan"))
    inv_db = -db if db == db else float("-inf")  # db==db tests for NaN
    if primary == "silhouette":
        return (sil, ch, inv_db)
    if primary == "db":
        return (inv_db, sil, ch)
    if primary == "ch":
        return (ch, sil, inv_db)
    return (sil, ch, inv_db)


def run_select(req: SelectRequest) -> SelectResponse:
    best: Tuple[float, float, float] = (float("-inf"), float("-inf"), float("-inf"))
    winner: TrainResponse | None = None
    for c in req.candidates:
        s = _score_tuple(c.metrics, req.primary_metric)
        if s > best:
            best = s
            winner = c
    assert winner is not None, "No candidates provided"
    rationale = f"Selected by {req.primary_metric} (tie-broken by CH and inverse DB)."
    return SelectResponse(winner=winner, rationale=rationale)