# ### FILE: service/tools/train.py
# from __future__ import annotations
# import numpy as np
# from typing import Dict, Any
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture

# try:
#     import hdbscan
#     HDBSCAN_AVAILABLE = True
# except Exception:
#     HDBSCAN_AVAILABLE = False

# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# from contracts.schemas import TrainRequest, TrainResponse
# from service.settings import settings
# from service.utils.io import ensure_dir, dump_object


# def _metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
#     out = {"silhouette": float("nan"), "db": float("nan"), "ch": float("nan")}
#     unique = np.unique(labels[labels >= 0]) if np.any(labels >= 0) else np.array([])
#     if unique.shape[0] < 2:
#         return out
#     mask = labels >= 0
#     Xs = X[mask] if np.any(mask) else X
#     ls = labels[mask] if np.any(mask) else labels
#     try:
#         out["silhouette"] = float(silhouette_score(Xs, ls))
#     except Exception:
#         pass
#     try:
#         out["db"] = float(davies_bouldin_score(Xs, ls))
#     except Exception:
#         pass
#     try:
#         out["ch"] = float(calinski_harabasz_score(Xs, ls))
#     except Exception:
#         pass
#     return out


# def run_train(req: TrainRequest) -> TrainResponse:
#     X_emb = np.load(req.X_emb_path)

#     if req.algorithm == "kmeans":
#         model = KMeans(**req.params).fit(X_emb)
#         labels = model.labels_
#     elif req.algorithm == "gmm":
#         model = GaussianMixture(**req.params).fit(X_emb)
#         labels = model.predict(X_emb)
#     elif req.algorithm == "hdbscan":
#         if not HDBSCAN_AVAILABLE:
#             raise RuntimeError("hdbscan not installed")
#         model = hdbscan.HDBSCAN(**req.params).fit(X_emb)
#         labels = model.labels_
#     else:
#         raise ValueError(f"Unsupported algorithm: {req.algorithm}")

#     m = _metrics(X_emb, labels)

#     ensure_dir(settings.models_dir)
#     model_path = f"{settings.models_dir}/model_{req.algorithm}.joblib"
#     labels_path = f"{settings.models_dir}/labels_{req.algorithm}.npy"

#     dump_object(model_path, model)
#     np.save(labels_path, labels)

#     return TrainResponse(
#         model_path=model_path,
#         labels_path=labels_path,
#         metrics=m,
#         algo=req.algorithm,
#         params=req.params,
#     )


### FILE: service/tools/train.py
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Type
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from contracts.schemas import TrainRequest, TrainResponse
from service.settings import settings
from service.utils.io import ensure_dir, dump_object


# -------------------------------
# Utility Functions
# -------------------------------
def compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering quality metrics safely."""
    metrics = {"silhouette": float("nan"), "db": float("nan"), "ch": float("nan")}

    valid_labels = labels >= 0
    if np.unique(labels[valid_labels]).shape[0] < 2:
        return metrics

    Xs, ls = X[valid_labels], labels[valid_labels]

    try:
        metrics["silhouette"] = float(silhouette_score(Xs, ls))
    except Exception:
        pass
    try:
        metrics["db"] = float(davies_bouldin_score(Xs, ls))
    except Exception:
        pass
    try:
        metrics["ch"] = float(calinski_harabasz_score(Xs, ls))
    except Exception:
        pass

    return metrics


# -------------------------------
# Abstract Trainer
# -------------------------------
class BaseTrainer(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return labels."""
        pass


# -------------------------------
# Concrete Trainers
# -------------------------------
class KMeansTrainer(BaseTrainer):
    def fit(self, X: np.ndarray) -> np.ndarray:
        self.model = KMeans(**self.params).fit(X)
        return self.model.labels_


class GMMTrainer(BaseTrainer):
    def fit(self, X: np.ndarray) -> np.ndarray:
        self.model = GaussianMixture(**self.params).fit(X)
        return self.model.predict(X)


class HDBSCANTrainer(BaseTrainer):
    def fit(self, X: np.ndarray) -> np.ndarray:
        if not HDBSCAN_AVAILABLE:
            raise RuntimeError("hdbscan is not installed")
        self.model = hdbscan.HDBSCAN(**self.params).fit(X)
        return self.model.labels_


# -------------------------------
# Factory
# -------------------------------
class TrainerFactory:
    _registry: Dict[str, Type[BaseTrainer]] = {
        "kmeans": KMeansTrainer,
        "gmm": GMMTrainer,
        "hdbscan": HDBSCANTrainer,
    }

    @classmethod
    def get_trainer(cls, algo: str, params: Dict[str, Any]) -> BaseTrainer:
        if algo not in cls._registry:
            raise ValueError(f"Unsupported algorithm: {algo}")
        return cls._registry[algo](params)


# -------------------------------
# Service
# -------------------------------
class ClusteringService:
    def __init__(self, req: TrainRequest):
        self.req = req
        self.X_emb = np.load(req.X_emb_path)
        self.trainer = TrainerFactory.get_trainer(req.algorithm, req.params)

    def run(self) -> TrainResponse:
        labels = self.trainer.fit(self.X_emb)
        metrics = compute_metrics(self.X_emb, labels)

        ensure_dir(settings.models_dir)
        model_path = f"{settings.models_dir}/model_{self.req.algorithm}.joblib"
        labels_path = f"{settings.models_dir}/labels_{self.req.algorithm}.npy"

        dump_object(model_path, self.trainer.model)
        np.save(labels_path, labels)

        return TrainResponse(
            model_path=model_path,
            labels_path=labels_path,
            metrics=metrics,
            algo=self.req.algorithm,
            params=self.req.params,
        )


# -------------------------------
# Entry Point
# -------------------------------
def run_train(req: TrainRequest) -> TrainResponse:
    service = ClusteringService(req)
    return service.run()
