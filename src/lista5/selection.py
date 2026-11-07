"""Selección hacia adelante (forward selection) por riesgo de entrenamiento."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .models import ajustar_y_evaluar


def seleccion_hacia_adelante(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    candidatas: Sequence[str],
    objetivo: str,
) -> pd.DataFrame:
    """Selección hacia adelante usando solo el riesgo de entrenamiento.

    En cada paso se agrega la variable que más disminuye el riesgo de
    entrenamiento, sin retirar las anteriores, hasta usar todas las candidatas.
    La validación solo se calcula para registrar el riesgo; no interviene en la
    elección de la variable agregada.

    Devuelve una fila por paso con: número de variables, variable agregada,
    riesgos de entrenamiento y validación, y las variables del modelo.
    """
    seleccionadas: list[str] = []
    restantes = list(candidatas)
    pasos = []

    while restantes:
        mejor = None
        for var in restantes:
            res = ajustar_y_evaluar(
                entrenamiento, validacion, seleccionadas + [var], objetivo
            )
            if mejor is None or res["riesgo_train"] < mejor[1]["riesgo_train"]:
                mejor = (var, res)
        var, res = mejor
        seleccionadas.append(var)
        restantes.remove(var)
        pasos.append(
            {
                "n_variables": len(seleccionadas),
                "variable_agregada": var,
                "riesgo_train": res["riesgo_train"],
                "riesgo_val": res["riesgo_val"],
                "variables": list(seleccionadas),
            }
        )

    return pd.DataFrame(pasos)
