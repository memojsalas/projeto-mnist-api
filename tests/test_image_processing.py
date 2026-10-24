"""Testes do pipeline de pré-processamento de imagem."""
import numpy as np
from PIL import Image
from app.image_processing import preprocess_image, load_image_from_bytes


def _fake_digit_image():
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[10:18, 12:16] = 255
    return Image.fromarray(arr, mode="L")


def test_preprocess_shape_and_range():
    out = preprocess_image(_fake_digit_image())
    assert out.shape == (28, 28, 1)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_load_invalid_bytes():
    try:
        load_image_from_bytes(b"nao-e-imagem")
        assert False
    except Exception:
        pass
