"""
Ponto de entrada da aplicação FastAPI.
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.config import settings
from app.logging_config import setup_logging
from app.routers import health, predict, predict_image, ui
from app.services import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    app.state.start_time = time.time()
    app.state.requests_total = 0

    app.state.model_service = ModelService(
        model_path=str(settings.model_path),
        preprocess_path=str(settings.preprocess_path),
        temperature_path=str(settings.temperature_path),
        ood_threshold_path=str(settings.ood_threshold_path),
        model_version=settings.model_version,
    )
    yield
    app.state.model_service = None


app = FastAPI(
    title="MNIST Classifier API",
    description="Classificação de dígitos manuscritos com OOD e calibração.",
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(predict_image.router)
app.include_router(ui.router)
app.mount("/metrics", make_asgi_app())


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "MNIST Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict_json": "/v1/predict",
        "predict_image": "/v1/predict/image",
        "predict_base64": "/v1/predict/base64",
        "metrics": "/metrics",
    }
