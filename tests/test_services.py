"""Testes unitários da camada de serviço."""
from pathlib import Path
from app.services import ModelService

MODELS = Path(__file__).resolve().parent.parent / "models"


def _service():
    return ModelService(
        model_path=str(MODELS / "mlp_mnist.keras"),
        preprocess_path=str(MODELS / "preprocess_pipeline.joblib"),
        temperature_path=str(MODELS / "temperature.json"),
        ood_threshold_path=str(MODELS / "ood_threshold.json"),
    )


def test_predict_returns_dict():
    service = _service()
    result = service.predict_from_pixels([0.0] * 784)
    assert len(result["probabilities"]) == 10
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-4
