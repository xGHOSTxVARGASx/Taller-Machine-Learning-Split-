"""Paquete lista5: utilidades reutilizables para la quinta lista de ejercicios de ML."""

from .data import cargar_hitters, particionar
from .metrics import correlaciones_con_objetivo, riesgo_cuadratico
from .models import ajustar_y_evaluar, pvalores_t
from .selection import seleccion_hacia_adelante

__all__ = [
    "cargar_hitters",
    "particionar",
    "correlaciones_con_objetivo",
    "riesgo_cuadratico",
    "ajustar_y_evaluar",
    "pvalores_t",
    "seleccion_hacia_adelante",
]

__version__ = "0.1.0"
