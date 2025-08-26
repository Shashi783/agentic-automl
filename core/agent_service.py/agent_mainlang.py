import asyncio
from contracts.schemas import ClusteringJobSpec, DatasetRef, FeatureSpec
from model_service.services.client import ModelServiceClient
from agent_graph import build_clustering_graph


async def main():
    svc = ModelServiceClient(base_url="http://localhost:8080")
    graph = build_clustering_graph(svc)

    spec = ClusteringJobSpec(
        dataset=DatasetRef(uri="file:///path/to/your/data.csv"),
        features=FeatureSpec(
            numeric=["feature1", "feature2", "feature3"],
            categorical=["country", "segment"],
            id_column="id",
        ),
        k_range=(2, 8),
        n_components=20,
        algorithms=["kmeans", "gmm", "hdbscan"],
        primary_metric="silhouette",
        scale_numeric=True,
        random_state=42,
    )

    final_state = await graph.ainvoke({"spec": spec})
    print("Final Registry:", final_state["registry"])

    await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
