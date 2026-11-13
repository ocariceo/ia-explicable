"""
Capítulo 05 - SHAP: Teoría de Juegos para entender características
TreeExplainer: optimizado para modelos basados en árboles.
"""
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from features import data

X = data.drop(columns=["approval_score", "approved"])
y = data["approved"]

forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)

explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X)

# Para clasificación binaria, el índice [:, :, 1] da los valores SHAP
# para la clase positiva
positive_class_shap = shap_values[:, :, 1]
importance = np.abs(positive_class_shap).mean(axis=0)
print(dict(zip(X.columns, importance.round(3))))
