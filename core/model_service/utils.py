### FILE: service/utils/io.py
from __future__ import annotations
import os
import json
import joblib
import pandas as pd
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def dump_object(path: str, obj: Any) -> None:
    joblib.dump(obj, path)


def load_object(path: str) -> Any:
    return joblib.load(path)


def load_dataframe(uri: str) -> pd.DataFrame:
    # In production, implement adapters: s3://, parquet, JDBC, etc.
    # For demo, support CSV/local file.
    if uri.startswith("file://"):
        path = uri.replace("file://", "")
        return pd.read_csv(path)
    if uri.endswith(".csv"):
        return pd.read_csv(uri)
    raise ValueError(f"Unsupported dataset uri: {uri}")