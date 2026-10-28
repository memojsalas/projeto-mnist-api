# 🧠 MNIST Classifier API

API REST para classificação de dígitos manuscritos (**MNIST**) com **calibração de probabilidades** (Temperature Scaling) e **detecção de anomalias (OOD)** utilizando **Energy Score**.

![Python](https://img.shields.io/badge/python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Resumo do Projeto

Este projeto resolve o problema de **classificação automática de dígitos manuscritos (0–9)** a partir de imagens 28×28 em escala de cinza. Além da classificação, a API sinaliza quando uma amostra está **fora da distribuição de treino (OOD)**, aumentando a confiabilidade em produção.

**Pipeline Híbrido:**

- **Treinamento:** `sklearn.Pipeline` + `SciKeras` (`KerasClassifier`) — permitindo o encapsulamento do fluxo e compatibilidade com o ecossistema Scikit-learn.
- **Produção:** modelo Keras nativo (`.keras`) + pipeline de pré-processamento (`.joblib`) carregados separadamente para inicialização rápida.

---

## 🧠 Origem do Modelo

- **Dataset:** MNIST (70.000 imagens 28×28, 10 classes).
- **Modelo:** MLP com 2 camadas ocultas (512 e 256 neurônios), Dropout 0.3.
- **Calibração:** Temperature Scaling (pós-treino) para ajustar a confiança do modelo.
- **Detecção OOD:** Energy Score (utilizando logits e threshold calculado no percentil 94).

**Métricas no teste (aproximadas):**

| Métrica | Valor |
| --- | --- |
| Acurácia Global | ~0,978 |
| Precisão Ponderada | ~0,978 |
| Recall Ponderado | ~0,978 |
| F1-Score Ponderado | ~0,978 |
| ROC-AUC (OvR) | ~0,999 |

---

## 🛠 Tecnologias

- **Python 3.10+** (compatível com 3.9 e 3.11)
- **TensorFlow 2.15 / Keras**
- **SciKeras 0.12**
- **FastAPI 0.104 + Pydantic 2.5**
- **Uvicorn** (ASGI)
- **Scikit-learn, NumPy, SciPy, Seaborn, Pillow, Joblib**
- **Prometheus** (métricas)

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
│   ├── services.py               # Camada de serviço (Energy Score + inferência)
│   └── routers/
│       ├── __init__.py           # Torna `app.routers` um pacote Python
│       ├── health.py             # GET /health
│       ├── predict_image.py      # POST /v1/predict/image (Único endpoint de inferência)
│       └── ui.py                 # Interface Web HTML/JS (GET /app)
│
├── models/                       # Artefatos do modelo (gerados pelo notebook)
│   ├── mlp_mnist.keras           # Modelo Keras treinado
│   ├── preprocess_pipeline.joblib# Pipeline de pré-processamento (sklearn)
│   ├── temperature.json          # Temperatura de calibração
│   ├── ood_threshold.json        # Threshold para Energy Score
│   ├── metrics.csv               # Métricas consolidadas
│   └── confusion_matrix.png      # Matriz de confusão
│
├── notebooks/                    # Notebook de treinamento
│   └── treinamento_mnist.ipynb   # Treina, calibra, avalia e exporta artefatos
│
├── data/                         # Dados e imagens de teste
│
├── tests/                        # Testes automatizados
│   ├── __init__.py               
│   ├── test_api.py               # Testa /health e /v1/predict/image
│   ├── test_services.py          # Testa a camada de serviço
│   └── test_image_processing.py  # Testa o pré-processamento de imagens
│
├── requirements.txt              # Dependências + restrição de Python
├── .gitignore                    # Regras de exclusão do Git
└──  Licence                      # Licence 
```

---

## 🚀 Instalação e Execução Local

### 1. Clonar o repositório

```bash
git clone https://github.com/memojsalas/projeto-mnist-api.git

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

### 4. Rodar o servidor localmente

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*(O servidor estará rodando no endereço `http://localhost:8000`)*
---

## 🖥️ Como Usar Localmente

Com o servidor rodando, você tem **duas formas** de interagir com o projeto:

### 1. Interface Web (Recomendado para testes visuais)

A API inclui uma interface gráfica amigável construída em HTML/CSS/JS.

1. Abra o seu navegador.
2. Acesse: **`http://localhost:8000/app`**
3. **Como usar:**
   - Clique na caixa tracejada ou arraste uma imagem (PNG/JPG) de um número.
   - O sistema processará a imagem e exibirá um **Card Verde** (se for um número válido) ou um **Card Amarelo** (se for detectado como OOD, ex: uma letra).
   - Clique em "Analisar outra imagem" para reiniciar o teste.

### 2. Swagger UI (Recomendado para testar a API REST)

O FastAPI gera uma documentação interativa automática. Como configuramos a API para ter apenas um endpoint de inferência, ele estará destacado.

1. Acesse: **`http://localhost:8000/docs`**
2. Abra o endpoint `POST /v1/predict/image`.
3. Clique em **"Try it out"**.
4. Faça o upload de uma imagem no campo `file`.
5. Clique em **"Execute"** e veja a resposta JSON.

---

## 📖 Endpoints da API

A API foi projetada para ter uma única rota de inferência, focada exclusivamente no upload de imagens.

| Endpoint | Método | Descrição |
| --- | --- | --- |
| `/` | GET | Informações gerais |
| `/health` | GET | Integridade do serviço e do modelo em memória |
| `/v1/predict/image` | POST | **Único endpoint de inferência (Upload de imagem)** |
| `/app` | GET | Interface Web interativa para testes |
| `/metrics` | GET | Métricas Prometheus |
| `/docs` | GET | Swagger UI |

### Exemplo de resposta JSON (`/v1/predict/image`)

**Se for um número válido (In-Distribution):**

```json
{
  "predicted_class": 7,
  "probabilities": [0.001, 0.002, "... 10 valores ..."],
  "confidence": 0.997,
  "is_ood": false,
  "message": "Dígito 7 reconhecido com sucesso."
}
```

**Se for uma imagem desconhecida (OOD):**

```json
{
  "predicted_class": "Desconhecido",
  "probabilities": [0.1, 0.05, "... 10 valores ..."],
  "confidence": 0.45,
  "is_ood": true,
  "message": "A imagem não parece ser um dígito válido de 0 a 9."
}
```
---

## 🧪 Testes

Para rodar os testes automatizados (que cobrem o modelo, o pré-processamento e as respostas da API):

```bash
pytest tests/ -v
```

---

## 📜 Licença

Este projeto está licenciado sob a **MIT License**.

---

## 👤 Autor

**Guillermo J Salas Avila**
📧 <memojsalas@gmail.com>
🔗 [GitHub](https://github.com/memojsalas)
