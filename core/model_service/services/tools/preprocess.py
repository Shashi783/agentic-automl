### FILE: service/tools/preprocess.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from contracts.schemas import PreprocessRequest, PreprocessResponse
from service.settings import settings
from service.utils.io import ensure_dir, dump_object, load_dataframe


def build_preprocessor(req: PreprocessRequest) -> Pipeline:
    numeric = req.features.numeric
    categorical = req.features.categorical

    num_steps = []
    if req.scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps) if num_steps else "passthrough"

    cat_pipe = Pipeline(steps=[
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if categorical else "passthrough"

    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_preprocess(req: PreprocessRequest) -> PreprocessResponse:
    df = load_dataframe(req.dataset.uri)

    X = df[req.features.numeric + req.features.categorical] if req.features.categorical else df[req.features.numeric]

    pre = build_preprocessor(req)
    Xp = pre.fit_transform(X)
    if isinstance(Xp, list):
        Xp = np.asarray(Xp)

    ensure_dir(settings.data_dir)
    X_path = f"{settings.data_dir}/X.npy"
    transformer_path = f"{settings.data_dir}/preprocessor.joblib"

    np.save(X_path, Xp)
    dump_object(transformer_path, pre)

    return PreprocessResponse(
        X_path=X_path,
        transformer_path=transformer_path,
        n_samples=Xp.shape[0],
        n_features=Xp.shape[1]
    )