"""
Métricas Prometheus.
"""
from prometheus_client import Counter, Histogram, Gauge

PREDICTIONS_TOTAL = Counter(
    "mnist_predictions_total",
    "Total de predições realizadas",
    ["endpoint", "predicted_class"],
)
PREDICTION_LATENCY = Histogram(
    "mnist_prediction_latency_seconds",
    "Latência da predição",
    ["endpoint"],
)
OOD_TOTAL = Counter("mnist_ood_total", "Total de amostras OOD")
REQUESTS_TOTAL = Gauge("mnist_requests_total", "Total de requisições")
