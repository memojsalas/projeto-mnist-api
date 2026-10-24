"""
Configurações da aplicação via variáveis de ambiente.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_dir: Path = Path(__file__).resolve().parent.parent
    model_path: Path = base_dir / "models" / "mlp_mnist.keras"
    preprocess_path: Path = base_dir / "models" / "preprocess_pipeline.joblib"
    temperature_path: Path = base_dir / "models" / "temperature.json"
    ood_threshold_path: Path = base_dir / "models" / "ood_threshold.json"

    max_image_size_mb: int = 5
    max_image_dimension: int = 4096

    binarize_threshold: int = 20
    digit_margin: int = 20

    model_version: str = "1.0.0"
    api_version: str = "1.0.0"
    log_level: str = "INFO"


settings = Settings()
