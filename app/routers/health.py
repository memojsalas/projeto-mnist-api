"""
Rota de monitoramento de saúde.
"""
import time
from fastapi import APIRouter, Request
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
async def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "model_service", None)
    start_time = getattr(request.app.state, "start_time", time.time())
    return HealthResponse(
        status="ok" if service is not None else "degraded",
        model_loaded=service is not None,
        model_version=service.model_version if service else "unknown",
        uptime_seconds=round(time.time() - start_time, 2),
        requests_total=getattr(request.app.state, "requests_total", 0),
    )
