"""
Pipeline de pré-processamento de imagens manuscritas para o formato MNIST.
"""
import io
import base64
import logging

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger(__name__)


class ImageProcessingError(ValueError):
    pass


def decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_string, validate=True)
    except Exception as e:
        raise ImageProcessingError(f"Base64 inválido: {e}")
    return load_image_from_bytes(raw)


def load_image_from_bytes(raw_bytes: bytes) -> Image.Image:
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise ImageProcessingError(
            f"Imagem excede o limite de {settings.max_image_size_mb} MB."
        )
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
        img = Image.open(io.BytesIO(raw_bytes))
    except (UnidentifiedImageError, Exception) as e:
        raise ImageProcessingError(f"Formato de imagem inválido: {e}")
    if max(img.size) > settings.max_image_dimension:
        raise ImageProcessingError(
            f"Dimensão máxima permitida: {settings.max_image_dimension}px."
        )
    return img


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img, dtype=np.uint8)

    arr = np.where(arr > settings.binarize_threshold, arr, 0).astype(np.uint8)

    ys, xs = np.where(arr > settings.binarize_threshold)
    if len(xs) == 0:
        raise ImageProcessingError("Nenhum dígito detectado na imagem.")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    digit = arr[y0:y1 + 1, x0:x1 + 1]

    h, w = digit.shape
    escala = settings.digit_margin / max(h, w)
    novo_h = max(1, int(round(h * escala)))
    novo_w = max(1, int(round(w * escala)))

    digit_img = Image.fromarray(digit).resize((novo_w, novo_h), Image.LANCZOS)
    digit_arr = np.array(digit_img)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - novo_h) // 2
    x_off = (28 - novo_w) // 2
    canvas[y_off:y_off + novo_h, x_off:x_off + novo_w] = digit_arr

    ys, xs = np.where(canvas > settings.binarize_threshold)
    if len(xs) > 0:
        cy = int(ys.mean())
        cx = int(xs.mean())
        canvas = np.roll(canvas, shift=(14 - cy, 14 - cx), axis=(0, 1))

    return (canvas.astype("float32") / 255.0).reshape(28, 28, 1)
