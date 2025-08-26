### FILE: service/tools/registry.py
from __future__ import annotations
import os
from typing import Dict, Any
from contracts.schemas import RegisterRequest, RegisterResponse
from service.settings import settings
from service.utils.io import ensure_dir, dump_json


def run_register(req: RegisterRequest) -> RegisterResponse:
    ensure_dir(settings.models_dir)
    model_uri = f"file://{req.winner.model_path}"
    card = {
        "algorithm": req.winner.algo,
        "params": req.winner.params,
        "metrics": req.winner.metrics,
        "dataset": req.dataset.model_dump(),
        "features": req.features.model_dump(),
        "extra": req.extra_meta,
    }
    card_path = os.path.join(settings.models_dir, "model_card.json")
    dump_json(card_path, card)
    return RegisterResponse(model_uri=model_uri, card_uri=f"file://{card_path}")
