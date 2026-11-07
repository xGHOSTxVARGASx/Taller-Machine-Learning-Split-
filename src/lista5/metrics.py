"""Riesgo cuadrático (función propia) y correlaciones."""

from __future__ import annotations

import numpy as np
import pandas as pd


def riesgo_cuadratico(y_real, y_pred) -> float:
    """Riesgo empírico con pérdida cuadrática: (1/|S|) * sum (y_i - h(x_i))^2."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_real - y_pred) ** 2))


def correlaciones_con_objetivo(df: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Correlación de Pearson de cada predictora con ``objetivo``.

    Devuelve un DataFrame ordenado de mayor a menor valor absoluto de la
    correlación, con las columnas ``correlacion`` y ``abs_correlacion``.
    """
    predictoras = [c for c in df.columns if c != objetivo]
    corr = df[predictoras].corrwith(df[objetivo])
    tabla = pd.DataFrame({"correlacion": corr, "abs_correlacion": corr.abs()})
    return tabla.sort_values("abs_correlacion", ascending=False)
