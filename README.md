# 🧠 MNIST Classifier API

API REST para classificação de dígitos manuscritos (**MNIST**) com **calibração de probabilidades** (Temperature Scaling) e **detecção de Out-of-Distribution (OOD)**.

![Python](https://img.shields.io/badge/python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Resumo do Projeto

Este projeto resolve o problema de **classificação automática de dígitos manuscritos (0–9)** a partir de imagens 28×28 em escala de cinza. Além da classificação, a API sinaliza quando uma amostra está **fora da distribuição de treino (OOD)**, aumentando a confiabilidade em produção.

**Pipeline híbrido:**

- **Treinamento:** `sklearn.Pipeline` + `SciKeras` (`KerasClassifier`) — permite `GridSearchCV`, `cross_val_score` e encapsulamento do fluxo.
- **Produção:** modelo Keras nativo (`.keras`) + pipeline de pré-processamento (`.joblib`) carregados **separadamente** para inicialização rápida.

---

## 🧠 Origem do Modelo

- **Dataset:** MNIST (70.000 imagens 28×28, 10 classes).
- **Modelo:** MLP com 2 camadas ocultas (512 e 256 neurônios), Dropout 0.3.
- **Calibração:** Temperature Scaling (pós-treino).
- **OOD:** Maximum Softmax Probability (MSP) com threshold no percentil 5.

**Métricas no teste (aproximadas):**

| Métrica | Valor |
|---|---|
| Acurácia Global | ~0,98 |
| Precisão Ponderada | ~0,98 |
| Recall Ponderado | ~0,98 |
| F1-Score Ponderado | ~0,98 |
| ROC-AUC (OvR) | ~0,999 |

---

## 🛠 Tecnologias

- **Python 3.10+** (compatível com 3.9 e 3.11)
- **TensorFlow 2.15 / Keras**
- **SciKeras 0.12**
- **FastAPI 0.104 + Pydantic 2.5**
- **Uvicorn** (ASGI)
- **Scikit-learn, NumPy, SciPy, Seaborn, Pillow**
- **Prometheus** (métricas)
- **Docker / Docker Compose** (opcional)

---

## 📁 Estrutura do Projeto

```
projeto-mnist-api/
│
├── app/                          # Código da API FastAPI
│   ├── __init__.py               # Torna `app` um pacote Python
│   ├── config.py                 # Configurações via variáveis de ambiente
│   ├── image_processing.py       # Pipeline de pré-processamento de imagens
│   ├── logging_config.py         # Logging estruturado em JSON
│   ├── main.py                   # Ponto de entrada (lifespan + routers)
│   ├── metrics.py                # Métricas Prometheus
│   ├── schemas.py                # Schemas Pydantic (validação estrita)
│   ├── services.py               # Camada de serviço (carrega modelo + inferência)
│   └── routers/
│       ├── __init__.py           # Torna `app.routers` um pacote Python
│       ├── health.py             # GET /health
│       ├── predict.py            # POST /v1/predict (JSON de 784 pixels)
│       └── predict_image.py      # POST /v1/predict/image e /v1/predict/base64
│
├── models/                       # Artefatos do modelo (gerados pelo notebook)
│   ├── mlp_mnist.keras           # Modelo Keras treinado
│   ├── preprocess_pipeline.joblib# Pipeline de pré-processamento (sklearn)
│   ├── temperature.json          # Temperatura de calibração (Temperature Scaling)
│   ├── ood_threshold.json        # Threshold para detecção OOD
│   ├── metrics.csv               # Métricas consolidadas
│   └── confusion_matrix.png      # Matriz de confusão
│
├── notebooks/                    # Notebook de treinamento
│   └── treinamento_mnist.ipynb   # Treina, calibra, avalia e exporta artefatos
│
├── data/                         # Dados e imagens de teste
│   └── .gitkeep                  # (opcional) imagens digitalizadas para teste
│
├── tests/                        # Testes automatizados
│   ├── __init__.py               # Torna `tests` um pacote Python
│   ├── test_api.py               # Testa /health e /v1/predict
│   ├── test_services.py          # Testa a camada de serviço
│   └── test_image_processing.py  # Testa o pré-processamento de imagens
│
├── requirements.txt              # Dependências + restrição de Python
├── Dockerfile                    # Build multi-stage para produção
├── docker-compose.yml            # API + Prometheus + Grafana
├── prometheus.yml                # Configuração de scrape do Prometheus
├── .env.example                  # Exemplo de variáveis de ambiente
├── .gitignore                    # Regras de exclusão do Git
├── .python-version               # Versão do Python (pyenv)
└── README.md                     # Este arquivo
```

### 🔍 Descrição dos módulos principais

| Módulo | Responsabilidade |
|---|---|
| `app/main.py` | Cria a aplicação FastAPI, configura `lifespan` (carrega o modelo uma única vez) e inclui os roteadores |
| `app/config.py` | Centraliza configurações (caminhos, limites de upload, thresholds) via `pydantic-settings` |
| `app/schemas.py` | Define os contratos de entrada e saída (Pydantic) com validação estrita |
| `app/services.py` | Carrega `.keras` + `.joblib` + `.json` e executa a inferência calibrada |
| `app/image_processing.py` | Converte fotos comuns (fundo claro, dígito escuro) no formato MNIST |
| `app/routers/` | Agrupa os endpoints (`/health`, `/v1/predict`, `/v1/predict/image`, `/v1/predict/base64`) |
| `app/metrics.py` | Contadores e histogramas Prometheus |
| `app/logging_config.py` | Logs em JSON, prontos para ELK/Datadog/CloudWatch |
| `tests/` | Testes unitários e de integração com `pytest` + `TestClient` |
| `notebooks/` | Treinamento, calibração, avaliação e exportação dos artefatos |

---

## 🚀 Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO
```

### 2. Criar o ambiente virtual (Python 3.10)

```bash
# Linux/Mac
python3.10 -m venv venv
source venv/bin/activate

# Windows
py -3.10 -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Rodar o servidor

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse a documentação interativa: **http://localhost:8000/docs**

---

## 📖 Documentação da API

| Endpoint | Método | Descrição |
|---|---|---|
| `/` | GET | Informações gerais |
| `/health` | GET | Integridade do serviço |
| `/v1/predict` | POST | Classifica a partir de 784 pixels (JSON) |
| `/v1/predict/image` | POST | Classifica a partir de upload de imagem |
| `/v1/predict/base64` | POST | Classifica a partir de imagem em base64 |
| `/metrics` | GET | Métricas Prometheus |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

### Exemplo de payload (`POST /v1/predict`)

```json
{
  "pixels": [0.0, 0.0, "... 784 valores (0-255) ...", 255.0]
}
```

### Exemplo de resposta

```json
{
  "predicted_class": 7,
  "probabilities": [0.001, 0.002, "... 10 valores ..."],
  "confidence": 0.997,
  "is_ood": false,
  "preprocessed_pixels": ["... 784 valores ..."]
}
```

### Exemplo com `curl`

```bash
# JSON (784 pixels)
curl -X POST "http://localhost:8000/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"pixels": [0.0, 0.0, "... 784 valores ..."]}'

# Upload de imagem
curl -X POST "http://localhost:8000/v1/predict/image" \
     -F "file=@data/digito_teste.png"

# Base64
curl -X POST "http://localhost:8000/v1/predict/base64" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "iVBORw0KGgo..."}'
```

### Exemplo com Python

```python
import requests

with open("data/digito_teste.png", "rb") as f:
    r = requests.post(
        "http://localhost:8000/v1/predict/image",
        files={"file": ("digito.png", f, "image/png")},
    )
print(r.json())
```

---

## 🐳 Docker

```bash
# Build e execução
docker compose up --build

# Apenas a API
docker build -t mnist-api .
docker run -p 8000:8000 mnist-api
```

Serviços disponíveis:

| Serviço | Porta | URL |
|---|---|---|
| API | 8000 | http://localhost:8000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin/admin) |

---

## 🧪 Testes

```bash
pytest tests/ -v
```

---

## 🎥 Vídeo de Demonstração

📹 **[Assista no Google Drive](COLE_O_LINK_AQUI)**

---

## 📜 Licença

Este projeto está licenciado sob a **MIT License**.

---

## 👤 Autor

**Seu Nome**
📧 seu.email@exemplo.com
🔗 [GitHub](https://github.com/SEU_USUARIO)
