"""
Capítulo 06 - LIME: Explicaciones Locales en Todas Partes
Datos tabulares.

Nota: requiere 'log_reg' ya entrenado (ver 02_modelos_lineales).
"""
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from features import data

X = data.drop(columns=["approval_score", "approved"])
y_cls = data["approved"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
log_reg = LogisticRegression().fit(X_scaled, y_cls)

instance = X.iloc[1]  # el solicitante que queremos explicar

explainer = LimeTabularExplainer(
    X.values, feature_names=X.columns, mode="classification"
)

explanation = explainer.explain_instance(
    instance.values, log_reg.predict_proba
)
explanation.as_pyplot_figure()
