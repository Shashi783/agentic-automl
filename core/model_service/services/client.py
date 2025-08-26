import httpx
from contracts.schemas import *

class ModelServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)

    async def preprocess(self, req: dict):
        return (await self.client.post("/preprocess", json=req)).json()

    async def feature_extract(self, req: dict):
        return (await self.client.post("/feature_extract", json=req)).json()

    async def train(self, req: dict):
        return (await self.client.post("/train", json=req)).json()

    async def select(self, req: dict):
        return (await self.client.post("/select", json=req)).json()

    async def register(self, req: dict):
        return (await self.client.post("/register", json=req)).json()

    async def close(self):
        await self.client.aclose()
