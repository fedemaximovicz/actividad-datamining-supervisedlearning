"""Regla de derivación de mensajes y sus indicadores de negocio.

Completá las dos funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_derivacion.py
"""
import numpy as np

REVISION = "revision_humana"


def derivar(proba, clases, umbral_fraude, umbral_confianza):
    """Decide el destino de cada mensaje a partir de las probabilidades del modelo.

    proba : array (n_mensajes, n_clases) con P(clase | mensaje); columnas en el orden de `clases`.
    clases : lista o array con los nombres de las clases (incluye "fraude").

    Regla, en este orden de prioridad:
    1. Si P(fraude) >= umbral_fraude, el destino es "fraude" (aunque no sea la clase más probable).
    2. Si no, y la probabilidad máxima >= umbral_confianza, el destino es la clase más probable.
    3. En cualquier otro caso, el destino es REVISION ("revision_humana").

    Devuelve un array de n_mensajes strings.
    """
    proba = np.asarray(proba)
    clases = np.asarray(clases)
    pred = proba.argmax(axis=1)
    destino = np.full(len(proba), REVISION, dtype=object)
    seguros = proba.max(axis=1) >= umbral_confianza
    destino[seguros] = clases[pred[seguros]]
    destino[proba[:, np.flatnonzero(clases == "fraude")[0]] >= umbral_fraude] = "fraude"
    return destino


def evaluar_derivacion(y_real, destino):
    """Calcula los indicadores del negocio. Devuelve un dict con tres claves:

    - "cobertura": proporción de mensajes derivados automáticamente (destino distinto de REVISION).
    - "exactitud_automatica": entre los derivados automáticamente, proporción con destino == y_real.
    - "fraude_perdido": entre los mensajes cuyo y_real es "fraude", proporción derivada automáticamente
      a un equipo que no es "fraude". Los enviados a revisión humana no cuentan como perdidos.

    Los tres valores son proporciones entre 0 y 1 (float).
    """
    y_real, destino = np.asarray(y_real), np.asarray(destino)
    automaticos = destino != REVISION
    fraude = y_real == "fraude"
    return {
        "cobertura": float(automaticos.mean()) if len(destino) else 0.0,
        "exactitud_automatica": float((destino[automaticos] == y_real[automaticos]).mean()) if automaticos.any() else 0.0,
        "fraude_perdido": float((automaticos & (destino != "fraude") & fraude).sum() / fraude.sum()) if fraude.any() else 0.0,
    }
