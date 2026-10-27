"""
Rota de inferência via JSON (784 pixels).
"""
from fastapi import APIRouter, HTTPException, Request
from app.schemas import PredictRequest, PredictResponse
from app.metrics import PREDICTIONS_TOTAL, OOD_TOTAL

router = APIRouter()


@router.post(
    "/v1/predict",
    response_model=PredictResponse,
    summary="Classifica a partir de 784 pixels (JSON)",
    tags=["Inferência"],
)
async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado.")

    try:
        result = service.predict_from_pixels(payload.pixels)
        PREDICTIONS_TOTAL.labels(
            endpoint="/v1/predict",
            predicted_class=str(result["predicted_class"]),
        ).inc()
        if result["is_ood"]:
            OOD_TOTAL.inc()
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro de validação: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
