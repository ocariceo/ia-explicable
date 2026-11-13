# IA Explicable con Python

Ejemplos de código del libro **"IA Explicable con Python: Interpretando modelos
con SHAP, LIME y más"**, de Oscar Cariceo Rivera.

Cada carpeta corresponde a un capítulo del libro y contiene los fragmentos de
código tal como aparecen en el texto, listos para ejecutarse sobre el mismo
conjunto de datos sintético de aprobación de hipotecas (`01_dataset/features.py`).

## Estructura

| Carpeta | Capítulo | Contenido |
|---|---|---|
| `01_dataset/` | Cap. 1 | Dataset sintético (`features.py`) usado en todo el libro |
| `02_modelos_lineales/` | Cap. 2 | Coeficientes de regresión lineal y logística |
| `03_arboles/` | Cap. 3 | Reglas de árbol de decisión e importancia de features en random forest |
| `04_importancia_permutacion/` | Cap. 4 | Importancia de permutación (agnóstica al modelo) |
| `05_shap/` | Cap. 5 | `TreeExplainer` y `KernelExplainer` de SHAP |
| `06_lime/` | Cap. 6 | LIME para datos tabulares y texto |
| `07_consistencia_fidelidad/` | Cap. 7 | Métricas de consistencia y fidelidad de una explicación |
| `08_clustering/` | Cap. 8 | Explicabilidad en modelos sin etiquetas (silueta, ARI) |
| `09_modelos_generativos/` | Cap. 9 | Cadena de pensamiento y autocoherencia en modelos de chat |

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

Cada script importa `data` desde `features.py`. Para ejecutarlos directamente,
copia o enlaza `01_dataset/features.py` junto al script, o ajusta tu
`PYTHONPATH`, por ejemplo:

```bash
PYTHONPATH=01_dataset python 02_modelos_lineales/entrenar_modelos.py
```

## Notas

- Algunos scripts (Capítulos 6 y 9) referencian variables como `sentiment_model`
  o funciones como `get_response()` que el libro deja como ejercicio del
  lector: debes aportar tu propio modelo de sentimiento o tu propia llamada a
  una API de modelo de chat.
- El dataset es sintético y fue generado con ayuda de IA para evitar sesgos y
  proteger información real.

## Licencia

Código de ejemplo liberado bajo licencia MIT. El texto del libro conserva
todos los derechos de su autor, Oscar Cariceo Rivera.
