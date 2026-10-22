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

---

## 🚀 Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO
