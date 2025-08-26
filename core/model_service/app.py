from fastapi import FastAPI
from .routers.model_service_api import router as model_service_router

app = FastAPI(title="model service API", version="1.0.0")

# # Include the routers
app.include_router(model_service_router, prefix="/model_service_api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the model service API!"}

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False,
        workers=1,
    )