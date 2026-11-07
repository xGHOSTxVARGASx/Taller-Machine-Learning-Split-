"""Carga de datos y partición train / validación / prueba."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Carpeta data/ en la raíz del repositorio (src/lista5/data.py -> ../../data)
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

OBJETIVO = "Salary"
SEMILLA = 31


def cargar_hitters(nombre: str = "hitters.csv") -> pd.DataFrame:
    """Carga el archivo Hitters y deja Salary más las 19 predictoras.

    El CSV trae una columna ``index`` (número de fila) que no es una variable
    del problema, así que se elimina si está presente.
    """
    ruta = _DATA_DIR / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}")
    df = pd.read_csv(ruta)
    if "index" in df.columns:
        df = df.drop(columns="index")
    if df.isna().any().any():
        raise ValueError("El archivo contiene valores faltantes")
    return df


def particionar(
    df: pd.DataFrame,
    n_desarrollo: int = 120,
    n_entrenamiento: int = 50,
    random_state: int = SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición en dos etapas con ``train_test_split``.

    1. Separa ``n_desarrollo`` jugadores para desarrollo y el resto para prueba.
    2. Divide desarrollo en ``n_entrenamiento`` para entrenamiento y el resto
       para validación.

    Devuelve ``(entrenamiento, validacion, prueba)``.
    """
    desarrollo, prueba = train_test_split(
        df, train_size=n_desarrollo, random_state=random_state
    )
    entrenamiento, validacion = train_test_split(
        desarrollo, train_size=n_entrenamiento, random_state=random_state
    )
    return entrenamiento, validacion, prueba
