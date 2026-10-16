"""Evaluación económica de la campaña de retención de Nodo Internet.

Completá las funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_negocio.py
"""
import numpy as np

# Supuestos acordados por Finanzas y Fidelización (ver el enunciado del caso)
SUPUESTOS = {
    "costo_contacto": 10_000,            # Gs. por cliente contactado
    "costo_descuento": 200_000,          # Gs. por cliente que acepta el descuento
    "valor_cliente_retenido": 1_000_000,  # Gs. de margen de un cliente que se iba a ir y se queda
    "acepta_si_se_iba": 0.40,            # proporción que acepta la oferta entre los que se iban a ir
    "acepta_si_se_quedaba": 0.60,        # proporción que acepta la oferta entre los que se iban a quedar
}


def resultado_por_contacto(s=SUPUESTOS):
    """Devuelve una tupla (si_se_iba, si_se_quedaba) con el resultado esperado, en guaraníes, de contactar a:
    - un cliente que se iba a ir (si acepta, se queda y deja su margen, menos el descuento), y
    - un cliente que se iba a quedar (si acepta, la empresa resigna el descuento).
    En los dos casos se paga el costo del contacto. No contactar tiene resultado 0.
    """
    si_se_iba = (
        s["acepta_si_se_iba"]
        * (s["valor_cliente_retenido"] - s["costo_descuento"])
        - s["costo_contacto"]
    )
    si_se_quedaba = (
        -s["acepta_si_se_quedaba"] * s["costo_descuento"]
        - s["costo_contacto"]
    )
    return float(si_se_iba), float(si_se_quedaba)


def probabilidad_de_equilibrio(s=SUPUESTOS):
    """Probabilidad de baja p a partir de la cual contactar tiene resultado esperado positivo:
    el valor de p que hace p · si_se_iba + (1 - p) · si_se_quedaba = 0.
    """
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)
    if si_se_iba == si_se_quedaba:
        raise ValueError("Los resultados por contacto no definen un equilibrio único.")
    return -si_se_quedaba / (si_se_iba - si_se_quedaba)


def ganancia_campania(y_real, contactar, s=SUPUESTOS):
    """Ganancia total, en guaraníes, de contactar a los clientes marcados con True en `contactar`,
    sabiendo quiénes se fueron realmente (`y_real`, 1 = se fue). Devuelve un float.
    """
    y = np.asarray(y_real)
    mascara = np.asarray(contactar, dtype=bool)
    if y.ndim != 1 or mascara.shape != y.shape:
        raise ValueError("y_real y contactar deben ser vectores de igual longitud.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_real debe contener únicamente 0 y 1.")
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)
    return float(np.where(y[mascara] == 1, si_se_iba, si_se_quedaba).sum())


def mejor_umbral(y_real, prob, s=SUPUESTOS, max_contactos=None):
    """Busca el umbral u que maximiza la ganancia al contactar a todos los clientes con prob >= u.

    - Los umbrales candidatos son los valores distintos de `prob`.
    - Si max_contactos no es None, se descartan los umbrales que contactan a más clientes.
    - Devuelve un dict {"umbral": u, "ganancia": g, "contactados": n}.
    - Si ningún umbral da ganancia positiva, devuelve {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}.
    """
    y, probabilidades = _validar_riesgos(y_real, prob)
    if max_contactos is not None:
        if not isinstance(max_contactos, (int, np.integer)) or max_contactos < 0:
            raise ValueError("max_contactos debe ser un entero no negativo.")
    sin_contactos = {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}
    if len(y) == 0:
        return sin_contactos

    # Ordenar una vez y evaluar al final de cada grupo de probabilidades iguales.
    # Así prob >= umbral incluye todos los empates y nunca rompe el cupo.
    orden = np.argsort(-probabilidades, kind="stable")
    riesgos = probabilidades[orden]
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)
    ganancias = np.cumsum(np.where(y[orden] == 1, si_se_iba, si_se_quedaba))
    finales = np.flatnonzero(np.r_[riesgos[:-1] != riesgos[1:], True])
    if max_contactos is not None:
        finales = finales[finales + 1 <= max_contactos]
    if len(finales) == 0:
        return sin_contactos
    # Ante igual ganancia, el primer candidato contacta a menos clientes.
    mejor = finales[np.argmax(ganancias[finales])]
    if ganancias[mejor] <= 0:
        return sin_contactos
    return {
        "umbral": float(riesgos[mejor]),
        "ganancia": float(ganancias[mejor]),
        "contactados": int(mejor + 1),
    }


def lift(y_real, prob, fraccion):
    """Tasa de positivos entre la `fraccion` de clientes con mayor prob, dividida por la tasa general.

    Ejemplo: lift(y, prob, 0.07) compara la tasa de baja del 7% de mayor riesgo con la tasa de toda la base.
    La cantidad de clientes del grupo es round(fraccion · n), con un mínimo de 1.
    """
    y, probabilidades = _validar_riesgos(y_real, prob)
    if not 0 < fraccion <= 1:
        raise ValueError("fraccion debe estar entre 0 (excluido) y 1.")
    if len(y) == 0 or y.mean() == 0:
        return 0.0
    cantidad = max(1, round(fraccion * len(y)))
    orden = np.argsort(-probabilidades, kind="stable")
    return float(y[orden[:cantidad]].mean() / y.mean())


def _validar_riesgos(y_real, prob):
    """Valida los vectores alineados de bajas binarias y probabilidades."""
    y = np.asarray(y_real)
    probabilidades = np.asarray(prob, dtype=float)
    if y.ndim != 1 or probabilidades.shape != y.shape:
        raise ValueError("y_real y prob deben ser vectores de igual longitud.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_real debe contener únicamente 0 y 1.")
    if not np.isfinite(probabilidades).all() or not (
        (probabilidades >= 0) & (probabilidades <= 1)
    ).all():
        raise ValueError("prob debe contener probabilidades finitas entre 0 y 1.")
    return y, probabilidades
