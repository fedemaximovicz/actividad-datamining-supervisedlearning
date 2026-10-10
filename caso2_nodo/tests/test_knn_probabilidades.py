"""Regresiones numéricas de las probabilidades usadas en las métricas."""
import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.knn import KNNClasificador


@pytest.mark.parametrize("pesos", ["uniforme", "distancia"])
def test_vecinos_de_una_clase_producen_probabilidades_validas(pesos):
    # Hay dos clases en entrenamiento, pero los 51 vecinos votan por la misma.
    # Sumar 51 pesos ya normalizados puede producir 1.0000000000000007.
    X = np.arange(1, 54, dtype=float).reshape(-1, 1)
    y = np.zeros(53, dtype=int)
    y[-1] = 1
    modelo = KNNClasificador(k=51, pesos=pesos).fit(X, y)

    probas = modelo.predict_proba([[0.0]])

    assert np.isfinite(probas).all()
    assert ((probas >= 0) & (probas <= 1)).all()
    np.testing.assert_array_equal(probas, [[1.0, 0.0]])
    assert np.isfinite(log_loss([0], probas, labels=[0, 1]))
