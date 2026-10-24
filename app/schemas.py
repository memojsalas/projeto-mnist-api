"""
Schemas Pydantic para validação estrita.
"""
from typing import List
from pydantic import BaseModel, Field

# CORREÇÃO: Adicionado model_config em todas as classes
class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    pixels: List[float] = Field(
        ..., min_length=784, max_length=784,
        description="Lista de 784 valores de pixel (0-255).",
    )


class PredictBase64Request(BaseModel):
    model_config = {"protected_namespaces": ()}
    image_base64: str = Field(
        ..., min_length=10,
        description="Imagem PNG/JPEG em base64.",
    )


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    predicted_class: int = Field(..., ge=0, le=9)
    probabilities: List[float] = Field(..., min_length=10, max_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ood: bool
    preprocessed_pixels: List[float] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    requests_total: int
