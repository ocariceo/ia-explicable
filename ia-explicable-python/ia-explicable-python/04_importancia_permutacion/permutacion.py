"""
Capítulo 04 - Importancia de Permutación
Técnica agnóstica al modelo: mezcla cada columna y mide cuánto cae el
rendimiento del modelo.
"""
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from features import data

X = data.drop(columns=["approval_score", "approved"])
y = data["approved"]

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000,
                     random_state=0).fit(X, y)

result = permutation_importance(
    mlp, X, y,
    n_repeats=30,  # mezclar cada columna 30 veces para un promedio estable
    random_state=0,
    scoring="accuracy"
)

print(result.importances_mean)
