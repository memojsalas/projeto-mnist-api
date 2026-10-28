"""Testes da API com FastAPI TestClient."""
import pytest
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    """
    Fixture que inicializa o TestClient usando 'with'.
    Isso força o FastAPI a rodar o 'lifespan', carregando o modelo na memória.
    """
    with TestClient(app) as c:
        yield c

def test_health(client):
    """Testa se o endpoint de saúde está online e o modelo carregado."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_predict_image_valid(client):
    """Testa o upload de uma imagem válida."""
    # Cria uma imagem falsa (fundo preto)
    img_array = np.zeros((28, 28), dtype=np.uint8)
    
    # CORREÇÃO: Desenha um "dígito" falso (uma linha branca vertical no meio)
    img_array[5:23, 13:15] = 255 
    
    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Envia para a API
    r = client.post(
        "/v1/predict/image",
        files={"file": ("teste.png", buf, "image/png")}
    )
    
    assert r.status_code == 200
    body = r.json()
    assert "predicted_class" in body
    assert "is_ood" in body
    assert "message" in body

def test_predict_image_invalid_type(client):
    """Testa se a API bloqueia arquivos que não são imagens (Erro 415)."""
    r = client.post(
        "/v1/predict/image",
        files={"file": ("teste.txt", b"texto qualquer", "text/plain")}
    )
    assert r.status_code == 415 # Unsupported Media Type

def test_predict_image_blank(client):
    """Testa se a API bloqueia uma imagem em branco/preto sem dígitos (Erro 422)."""
    # Cria uma imagem totalmente preta
    img_array = np.zeros((28, 28), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post(
        "/v1/predict/image",
        files={"file": ("em_branco.png", buf, "image/png")}
    )
    assert r.status_code == 422 # A API deve rejeitar imagem sem dígitos