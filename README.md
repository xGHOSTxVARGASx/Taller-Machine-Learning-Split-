# Quinta lista de ejercicios — Introducción al Machine Learning

Selección de variables y generalización.
Universidad El Bosque · Programa de Estadística · Semestre 2026–2
**Autor:** Daniel Vargas

Se predice el salario (`Salary`, en miles de dólares) de 263 jugadores de béisbol (datos *Hitters*)
con regresión lineal con intercepto, comparando cuatro modelos: todas las variables, las siete de mayor
correlación, las variables significativas en pruebas t y selección hacia adelante. Los modelos se ajustan
solo con entrenamiento, se comparan con validación y se evalúan al final en prueba.

## Estructura

```
lista_5/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- data/
|   `-- hitters.csv
|-- src/
|   `-- lista5/
|       |-- __init__.py
|       |-- data.py           # carga de datos y partición (random_state=31)
|       |-- metrics.py        # riesgo cuadrático (función propia) y correlaciones
|       |-- models.py         # ajuste/evaluación de la regresión lineal y valores p (prueba t)
|       `-- selection.py      # selección hacia adelante
`-- notebooks/
    `-- lista_5.ipynb         # ejercicios 1 a 4: tablas, gráfica y respuestas
```

### Partición de los datos

`train_test_split` con `random_state=31`, en dos etapas: 120 jugadores de desarrollo y 143 de prueba;
después, los 120 de desarrollo se dividen en 50 de entrenamiento y 70 de validación.
La columna `index` del CSV (número de fila) se elimina al cargar los datos; quedan `Salary` y las 19 predictoras.

### Funciones principales

| Módulo | Función | Descripción |
|---|---|---|
| `data.py` | `cargar_hitters()`, `particionar()` | Lectura del CSV y partición entrenamiento / validación / prueba |
| `metrics.py` | `riesgo_cuadratico()`, `correlaciones_con_objetivo()` | Riesgo empírico cuadrático y correlaciones ordenadas por valor absoluto |
| `models.py` | `ajustar_y_evaluar()`, `pvalores_t()` | Regresión lineal con riesgos por conjunto y valores p de la prueba t de cada coeficiente |
| `selection.py` | `seleccion_hacia_adelante()` | Selección hacia adelante por riesgo de entrenamiento |

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone <URL_DEL_REPOSITORIO>
cd lista_5

python -m venv .venv
source .venv/bin/activate        # Git Bash en Windows: source .venv/Scripts/activate

pip install -r requirements.txt
pip install -e .                 # instala el paquete local lista5
```

## Cómo ejecutar el cuaderno

```bash
jupyter notebook notebooks/lista_5.ipynb
```

Use *Kernel → Restart & Run All*. El cuaderno se ejecuta completo desde un kernel reiniciado y lee
los datos automáticamente desde `data/`.

## Resumen de resultados

| Modelo | Variables | Riesgo train | Riesgo validación | Riesgo prueba |
|---|---|---|---|---|
| Todas las variables | 19 | 68 584.8 | 320 423.2 | 394 308.7 |
| Mayor correlación | CRBI, CRuns, CHmRun, CHits, CAtBat, CWalks, Years | 115 487.2 | 133 035.6 | 145 923.3 |
| Pruebas t | AtBat, CWalks, Assists | 161 221.4 | 140 085.2 | 158 772.5 |
| Selección hacia adelante | CRBI, Years, CRuns | 126 982.3 | 130 618.2 | 126 939.6 |

El modelo elegido con validación (selección hacia adelante) también fue el de menor riesgo de prueba.
El modelo completo tiene el menor riesgo de entrenamiento pero el mayor de prueba (sobreajuste).
