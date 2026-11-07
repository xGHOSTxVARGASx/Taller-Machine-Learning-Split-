"""Ajuste de regresión lineal, evaluación de riesgos y valores p."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from .metrics import riesgo_cuadratico


def ajustar_y_evaluar(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    variables: Sequence[str],
    objetivo: str,
    prueba: pd.DataFrame | None = None,
) -> dict:
    """Ajusta una regresión lineal con intercepto solo con ``entrenamiento``.

    Devuelve los riesgos de entrenamiento y validación (y de prueba si se
    entrega ``prueba``), junto con el modelo ajustado.
    """
    variables = list(variables)
    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(entrenamiento[variables], entrenamiento[objetivo])

    resultado = {
        "variables": variables,
        "n_variables": len(variables),
        "riesgo_train": riesgo_cuadratico(
            entrenamiento[objetivo], modelo.predict(entrenamiento[variables])
        ),
        "riesgo_val": riesgo_cuadratico(
            validacion[objetivo], modelo.predict(validacion[variables])
        ),
        "modelo": modelo,
    }
    if prueba is not None:
        resultado["riesgo_prueba"] = riesgo_cuadratico(
            prueba[objetivo], modelo.predict(prueba[variables])
        )
    return resultado


def pvalores_t(
    df: pd.DataFrame, predictoras: Sequence[str], objetivo: str
) -> pd.DataFrame:
    """Prueba t de cada coeficiente de la regresión lineal por MCO.

    Calcula a mano el estimador beta = (X'X)^-1 X'y, el error estándar
    sqrt(sigma^2 * diag((X'X)^-1)) con sigma^2 = RSS / (n - p - 1), el
    estadístico t y el valor p bilateral con la distribución t de Student.
    El intercepto se reporta en la fila ``intercepto``.
    """
    predictoras = list(predictoras)
    X = np.column_stack([np.ones(len(df)), df[predictoras].to_numpy(dtype=float)])
    y = df[objetivo].to_numpy(dtype=float)
    n, k = X.shape  # k = p + 1 parámetros

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuos = y - X @ beta
    gl = n - k
    sigma2 = float(residuos @ residuos) / gl
    error_est = np.sqrt(sigma2 * np.diag(xtx_inv))
    t = beta / error_est
    p = 2 * stats.t.sf(np.abs(t), df=gl)

    return pd.DataFrame(
        {"coeficiente": beta, "error_est": error_est, "t": t, "valor_p": p},
        index=["intercepto"] + predictoras,
    )
