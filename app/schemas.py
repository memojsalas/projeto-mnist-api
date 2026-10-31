"""
Schemas Pydantic para validação estrita.
"""
from typing import List, Union
from pydantic import BaseModel, Field

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
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_class": 7,
                    "probabilities": [0.001, 0.002, 0.003, 0.004, 0.005, 0.010, 0.020, 0.950, 0.003, 0.002],
                    "confidence": 0.950,
                    "is_ood": False,
                    "message": "Dígito 7 reconhecido com sucesso."
                }
            ]
        }
    }
    
    predicted_class: Union[int, str] = Field(
        ..., 
        description="Dígito previsto (0-9) ou 'Desconhecido' se a imagem for Out-of-Distribution."
    )
    
    probabilities: List[float] = Field(
        ..., 
        min_length=10, 
        max_length=10,
        description="Probabilidades calibradas para as 10 classes.",
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confiança máxima da previsão.",
    )
    
    is_ood: bool = Field(
        ..., 
        description="True se a imagem for considerada fora da distribuição (OOD)."
    )
    
    message: str = Field(
        default="", 
        description="Mensagem descritiva sobre o resultado da predição."
    )
    
    preprocessed_pixels: List[float] = Field(
        default_factory=list,
        description="Imagem 28x28 (784 valores) após o pré-processamento. Útil para debug.",
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    requests_total: int
