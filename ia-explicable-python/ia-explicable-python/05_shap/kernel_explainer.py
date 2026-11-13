"""
Capítulo 05 - SHAP: Teoría de Juegos para entender características
KernelExplainer: funciona en cualquier modelo, tratándolo como caja negra.

Nota: requiere que 'mlp' ya esté entrenado (ver 04_importancia_permutacion).
"""
import shap

from features import data
from sklearn.neural_network import MLPClassifier

X = data.drop(columns=["approval_score", "approved"])
y = data["approved"]

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000,
                     random_state=0).fit(X, y)

background = shap.sample(X, 5)  # conjunto de referencia pequeño, lo mantiene rápido
kernel = shap.KernelExplainer(mlp.predict_proba, background)
k_values = kernel.shap_values(X.iloc[[0]])
print(k_values)
