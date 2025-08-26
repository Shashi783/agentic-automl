from pydantic import BaseModel
from typing import List, Dict, Any

### FILE: contracts/schemas.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class APIResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    ok: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

############################################
# Data Models #
############################################







############################################
# Request Response Models #
############################################

# ------------------------ Common ------------------------
class DatasetRef(BaseModel):
    uri: str = Field(..., description="Dataset identifier (s3://, gs://, db://, file://)")


class FeatureSpec(BaseModel):
    numeric: List[str] = Field(default_factory=list)
    categorical: List[str] = Field(default_factory=list)
    id_column: Optional[str] = None


# ------------------------ Preprocess ------------------------
class PreprocessRequest(BaseModel):
    dataset: DatasetRef
    features: FeatureSpec
    scale_numeric: bool = True


class PreprocessResponse(BaseModel):
    X_path: str
    transformer_path: str
    n_samples: int
    n_features: int


# ------------------------ Feature Extraction ------------------------
class FeatureExtractRequest(BaseModel):
    X_path: str
    method: Literal["pca"] = "pca"
    n_components: int = 20
    random_state: int = 42


class FeatureExtractResponse(BaseModel):
    X_emb_path: str
    n_components: int


# ------------------------ Training ------------------------
AlgoName = Literal["kmeans", "gmm", "hdbscan"]


class TrainRequest(BaseModel):
    X_emb_path: str
    algorithm: AlgoName
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    model_path: str
    labels_path: str
    metrics: Dict[str, float]
    algo: AlgoName
    params: Dict[str, Any]


# ------------------------ Selection ------------------------
class SelectRequest(BaseModel):
    candidates: List[TrainResponse]
    primary_metric: Literal["silhouette", "db", "ch"] = "silhouette"


class SelectResponse(BaseModel):
    winner: TrainResponse
    rationale: str


# ------------------------ Registry ------------------------
class RegisterRequest(BaseModel):
    winner: TrainResponse
    dataset: DatasetRef
    features: FeatureSpec
    extra_meta: Dict[str, Any] = Field(default_factory=dict)


class RegisterResponse(BaseModel):
    model_uri: str
    card_uri: str


# ------------------------ Orchestration (Agent) ------------------------
class ClusteringJobSpec(BaseModel):
    dataset: DatasetRef
    features: FeatureSpec
    k_range: Tuple[int, int] = (2, 10)
    n_components: int = 20
    algorithms: List[AlgoName] = Field(default_factory=lambda: ["kmeans", "gmm"])  # hdbscan optional
    primary_metric: Literal["silhouette", "db", "ch"] = "silhouette"
    scale_numeric: bool = True
    random_state: int = 42