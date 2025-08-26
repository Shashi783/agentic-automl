from __future__ import annotations
from fastapi import APIRouter, HTTPException

from contracts.schemas import (
    APIResponse,
    PreprocessRequest, PreprocessResponse,
    FeatureExtractRequest, FeatureExtractResponse,
    TrainRequest, TrainResponse,
    SelectRequest, SelectResponse,
    RegisterRequest, RegisterResponse,
)

from ..services.tools.preprocess import run_preprocess
from ..services.tools.feature_extract import run_feature_extract
from ..services.tools.train import run_train
from ..services.tools.selector_service import run_select
from ..services.tools.registry import run_register

router = APIRouter()

@router.post("/tools/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest):
    try:
        return run_preprocess(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/feature_extract", response_model=FeatureExtractResponse)
def feature_extract(req: FeatureExtractRequest):
    try:
        return run_feature_extract(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        return run_train(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/select", response_model=SelectResponse)
def select(req: SelectRequest):
    try:
        return run_select(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    try:
        return run_register(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))