### FILE: service/tools/feature_extract.py
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from contracts.schemas import FeatureExtractRequest, FeatureExtractResponse
from service.settings import settings
from service.utils.io import ensure_dir


def run_feature_extract(req: FeatureExtractRequest) -> FeatureExtractResponse:
    X = np.load(req.X_path)
    if req.method == "pca":
        pca = PCA(n_components=req.n_components, random_state=req.random_state)
        X_emb = pca.fit_transform(X)
    else:
        raise ValueError(f"Unsupported method: {req.method}")

    ensure_dir(settings.data_dir)
    X_emb_path = f"{settings.data_dir}/X_emb.npy"
    np.save(X_emb_path, X_emb)

    return FeatureExtractResponse(X_emb_path=X_emb_path, n_components=X_emb.shape[1])