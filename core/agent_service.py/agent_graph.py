# FILE: agent/graph_lang.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from agent.client import ModelServiceClient
from contracts.schemas import ClusteringJobSpec


class PipelineState(TypedDict):
    spec: ClusteringJobSpec
    preprocess: Dict[str, Any]
    features: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    selection: Dict[str, Any]
    registry: Dict[str, Any]
    algo_index: int


def build_clustering_graph(svc: ModelServiceClient):
    """Build LangGraph pipeline for clustering job."""

    graph = StateGraph(PipelineState)

    # 1. Preprocess
    async def preprocess_node(state: PipelineState):
        resp = await svc.preprocess(state["spec"].dataset.model_dump())
        state["preprocess"] = resp
        return state

    graph.add_node("preprocess", preprocess_node)
    graph.set_entry_point("preprocess")

    # 2. Feature Extraction
    async def feature_node(state: PipelineState):
        resp = await svc.feature_extract({
            "dataset": state["spec"].dataset.model_dump(),
            "features": state["spec"].features.model_dump(),
            "scale_numeric": state["spec"].scale_numeric,
        })
        state["features"] = resp
        state["candidates"] = []
        state["algo_index"] = 0
        return state

    graph.add_node("features", feature_node)
    graph.add_edge("preprocess", "features")

    # 3. Training loop
    async def train_node(state: PipelineState):
        algos = state["spec"].algorithms
        algo = algos[state["algo_index"]]

        tr = await svc.train({
            "algorithm": algo,
            "params": {"random_state": state["spec"].random_state},
        })
        state["candidates"].append(tr)

        # advance index
        state["algo_index"] += 1
        return state

    graph.add_node("train", train_node)

    # Conditional: if more algos left → loop, else → select
    def train_router(state: PipelineState):
        if state["algo_index"] < len(state["spec"].algorithms):
            return "train"
        else:
            return "select"

    graph.add_edge("features", "train")
    graph.add_conditional_edges("train", train_router)

    # 4. Selection
    async def select_node(state: PipelineState):
        sel = await svc.select({
            "candidates": state["candidates"],
            "primary_metric": state["spec"].primary_metric,
        })
        state["selection"] = sel
        return state

    graph.add_node("select", select_node)

    # 5. Register
    async def register_node(state: PipelineState):
        reg = await svc.register({
            "winner": state["selection"]["winner"],
            "dataset": state["spec"].dataset.model_dump(),
            "features": state["spec"].features.model_dump(),
            "extra_meta": {"selection_rationale": state["selection"]["rationale"]},
        })
        state["registry"] = reg
        return state

    graph.add_node("register", register_node)
    graph.add_edge("select", "register")
    graph.set_finish_point("register")

    return graph.compile()
