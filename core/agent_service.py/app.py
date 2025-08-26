from fastapi import FastAPI
from contracts.schemas import ClusteringJobSpec
from model_service.services.client import ModelServiceClient
from agent_graph import build_clustering_graph

app = FastAPI()


@app.post("/invoke")
async def invoke_agent(spec: ClusteringJobSpec):
    """Invoke the clustering agent with a job specification."""
    svc = ModelServiceClient(base_url="http://localhost:8080")
    graph = build_clustering_graph(svc)

    final_state = await graph.ainvoke({"spec": spec})

    await svc.close()

    return {"registry": final_state["registry"]}
