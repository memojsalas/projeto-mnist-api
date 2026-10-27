"""
Rota de inferência via upload de imagem ou base64.
"""
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.image_processing import (
    ImageProcessingError,
    decode_base64_image,
    load_image_from_bytes,
)
from app.schemas import PredictBase64Request, PredictResponse
from app.metrics import PREDICTIONS_TOTAL, OOD_TOTAL

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/v1/predict/image",
    response_model=PredictResponse,
    summary="Classifica a partir de upload de imagem",
    tags=["Inferência"],
)
async def predict_from_upload(
    request: Request,
    file: UploadFile = File(..., description="PNG/JPEG da imagem manuscrita."),
) -> PredictResponse:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado.")

    allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type não suportado: {file.content_type}.",
        )

    try:
        raw = await file.read()
        img = load_image_from_bytes(raw)
        result = service.predict_from_image(img)

        PREDICTIONS_TOTAL.labels(
            endpoint="/v1/predict/image",
            predicted_class=str(result["predicted_class"]),
        ).inc()
        if result["is_ood"]:
            OOD_TOTAL.inc()

        return PredictResponse(**result)

    except ImageProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado na inferência")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")


@router.post(
    "/v1/predict/base64",
    response_model=PredictResponse,
    summary="Classifica a partir de imagem em base64",
    tags=["Inferência"],
)
async def predict_from_base64(
    request: Request,
    payload: PredictBase64Request,
) -> PredictResponse:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado.")

    try:
        img = decode_base64_image(payload.image_base64)
        result = service.predict_from_image(img)

        PREDICTIONS_TOTAL.labels(
            endpoint="/v1/predict/base64",
            predicted_class=str(result["predicted_class"]),
        ).inc()
        if result["is_ood"]:
            OOD_TOTAL.inc()

        return PredictResponse(**result)

    except ImageProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado na inferência (base64)")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
