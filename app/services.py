"""
Camada de serviço: carregamento do modelo e inferência com Energy Score OOD.
"""
import json
import logging

import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

from app.image_processing import preprocess_image

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(
        self,
        model_path: str,
        preprocess_path: str,
        temperature_path: str,
        ood_threshold_path: str,
        model_version: str = "1.0.0",
    ):
        logger.info("Carregando modelo Keras de %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Carregando pipeline de pré-processamento de %s", preprocess_path)
        self.preprocess = joblib.load(preprocess_path)
        self.model_version = model_version

        with open(temperature_path, "r") as f:
            self.temperature = float(json.load(f)["temperature"])
        with open(ood_threshold_path, "r") as f:
            self.ood_threshold = float(json.load(f)["threshold"])

        logits_layer = self.model.get_layer("logits")
        self.logits_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=logits_layer.output,
        )
        logger.info(
            "Modelo carregado. T=%.4f, Energy Threshold=%.4f",
            self.temperature, self.ood_threshold,
        )

    def _infer_from_array(self, arr_28x28x1: np.ndarray) -> dict:
        flat = arr_28x28x1.reshape(1, -1)
        flat_prep = self.preprocess.transform(flat)

        logits = self.logits_model.predict(flat_prep, verbose=0)
        probs = tf.nn.softmax(logits / self.temperature, axis=-1).numpy()[0]
        energy = -self.temperature * tf.reduce_logsumexp(logits / self.temperature, axis=-1).numpy()[0]

        is_ood = bool(energy > self.ood_threshold)

        # ALTERAÇÃO: Regra de negócio para OOD com mensagem amigável
        if is_ood:
            return {
                "predicted_class": "Desconhecido",
                "probabilities": [float(p) for p in probs],
                "confidence": float(np.max(probs)),
                "is_ood": True,
                "message": "A imagem não parece ser um dígito válido de 0 a 9."
            }

        pred_class = int(np.argmax(probs))
        return {
            "predicted_class": pred_class,
            "probabilities": [float(p) for p in probs],
            "confidence": float(np.max(probs)),
            "is_ood": False,
            "message": f"Dígito {pred_class} reconhecido com sucesso."
        }

    def predict_from_pixels(self, pixels: list) -> dict:
        arr = np.asarray(pixels, dtype=np.float32).reshape(28, 28, 1)
        if arr.max() > 1.0:
            arr = arr / 255.0
        result = self._infer_from_array(arr)
        result["preprocessed_pixels"] = (arr.squeeze() * 255.0).flatten().tolist()
        return result

    def predict_from_image(self, img: Image.Image) -> dict:
        arr = preprocess_image(img)
        result = self._infer_from_array(arr)
        result["preprocessed_pixels"] = (arr.squeeze() * 255.0).flatten().tolist()
        return result
