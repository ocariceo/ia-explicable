"""
Capítulo 07 - ¿Es Buena la Explicación?
Consistencia: ¿se mantiene la explicación a través de divisiones de datos?
"""
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap

from features import data

X = data.drop(columns=["approval_score", "approved"])
y = data["approved"]

X1, y1 = X.iloc[:4], y.iloc[:4]
X2, y2 = X.iloc[4:], y.iloc[4:]

model1 = RandomForestClassifier(random_state=0).fit(X1, y1)
model2 = RandomForestClassifier(random_state=0).fit(X2, y2)

imp1 = np.abs(shap.TreeExplainer(model1).shap_values(X1)[:, :, 1]).mean(axis=0)
imp2 = np.abs(shap.TreeExplainer(model2).shap_values(X2)[:, :, 1]).mean(axis=0)

consistency = cosine_similarity([imp1], [imp2])[0][0]
print(consistency)  # Salida: 0.94
