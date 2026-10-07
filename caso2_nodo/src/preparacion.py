"""Preparación de los datos de clientes de Nodo Internet.

Completá las dos funciones con pandas y la biblioteca estándar.
Verificá tu trabajo con:  python -m pytest -q tests/test_preparacion.py
"""
import unicodedata
import re

import numpy as np
import pandas as pd

METODOS_VALIDOS = ["debito_automatico", "tarjeta_credito", "transferencia", "efectivo"]
DIAS_POR_MES = 30.44


def normalizar_metodo_pago(valor):
    """Lleva las variantes con que el CRM guarda el medio de pago a uno de METODOS_VALIDOS.

    Ejemplos: "Débito automático", "DEBITO AUTOMATICO" y "Debito_Automatico" -> "debito_automatico";
    "Tarjeta de crédito" y "tarjeta credito" -> "tarjeta_credito"; "transferencia bancaria" -> "transferencia";
    "efectivo " -> "efectivo".
    Si el valor no corresponde a ninguno de los cuatro, lanzar ValueError.
    """
    #raise NotImplementedError("Completar normalizar_metodo_pago")
    valor = valor.lower()
    valor = valor.strip()

    valor = unicodedata.normalize('NFD', valor)
    valor = ''.join(c for c in valor if unicodedata.category(c) != 'Mn')

    valor = re.sub(r'[\s\-_]+', '_', valor)


    if 'debito' in valor:
        return 'debito_automatico'
    elif 'tarjeta' in valor or 'credito' in valor:
        return 'tarjeta_credito'
    elif 'transferencia' in valor:
        return 'transferencia'
    elif valor == 'efectivo':
        return 'efectivo'
    else:
        raise ValueError(
            f"El método de pago '{valor}' no es válido. Debe ser uno de {METODOS_VALIDOS}"
        )

    return valor


def agregar_variables(df):
    """Devuelve una COPIA de df (sin modificar el original) con metodo_pago normalizado y tres columnas nuevas.

    Todas se calculan a la fecha_corte de cada fila, nunca a la fecha de hoy:
    - antiguedad_meses: días entre fecha_alta y fecha_corte, divididos por DIAS_POR_MES.
    - meses_hasta_fin_permanencia: días entre fecha_corte y fin_permanencia, divididos por DIAS_POR_MES.
      Vale 0 en los contratos mensuales (no tienen fecha de fin de permanencia).
    - sin_permanencia_3m: 1 si el cliente puede irse sin penalidad en los próximos 3 meses, es decir,
      si su contrato es mensual o si meses_hasta_fin_permanencia <= 3; 0 en otro caso (entero).
    """

    #raise NotImplementedError("Completar agregar_variables")
    df_processed = df.copy()
    df_processed['metodo_pago'] = df_processed['metodo_pago'].apply(normalizar_metodo_pago)

    DIAS_POR_MES = 30.4375

    # 2. Asegurarse de convertir ambas columnas a formato datetime
    df_processed['fecha_alta'] = pd.to_datetime(df_processed['fecha_alta'])
    df_processed['fecha_corte'] = pd.to_datetime(df_processed['fecha_corte'])
    df_processed['fin_permanencia'] = pd.to_datetime(df_processed['fin_permanencia'])

    # 3. Restar las fechas, extraer los días con .dt.days y dividir
    df_processed['antiguedad_meses'] = (
        (df_processed['fecha_corte'] - df_processed['fecha_alta']).dt.days / DIAS_POR_MES
    )

    df_processed['meses_hasta_fin_permanencia'] = np.where(
        df_processed['tipo_contrato'] == 'mensual',0, \
          (df_processed['fecha_corte'] - df_processed['fecha_alta']).dt.days / DIAS_POR_MES
    )

    df_processed['meses_hasta_fin_permanencia'] = (
        df_processed['meses_hasta_fin_permanencia'].fillna(0).clip(lower=0)
    )

    df_processed['sin_permanencia_3m'] = np.where(
        (df_processed['tipo_contrato'] == 'mensual')
        | (df_processed['meses_hasta_fin_permanencia'] <= 3),
        1,
        0,
    )

    return df_processed
