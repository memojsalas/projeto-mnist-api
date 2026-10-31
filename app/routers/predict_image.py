"""
Rota de inferência via upload de imagem.
"""
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.image_processing import (
    ImageProcessingError,
    load_image_from_bytes,
)
from app.schemas import PredictResponse
from app.metrics import PREDICTIONS_TOTAL, OOD_TOTAL

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/v1/predict/image",
    response_model=PredictResponse,
    summary="Classifica a partir de upload de imagem",
    description="Envie uma imagem PNG ou JPEG contendo um dígito manuscrito (0-9). O sistema também detectará se a imagem não for um número (OOD).",
    tags=["Inferência"],
    responses={
        200: {"description": "Predição realizada com sucesso (Dígito válido ou OOD)."},
        415: {"description": "Tipo de arquivo não suportado. Envie apenas PNG ou JPEG."},
        422: {"description": "Imagem inválida ou nenhum dígito detectado na imagem."},
    }
)
async def predict_from_upload(
    request: Request,
    file: UploadFile = File(..., description="Arquivo PNG/JPEG da imagem manuscrita."),
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