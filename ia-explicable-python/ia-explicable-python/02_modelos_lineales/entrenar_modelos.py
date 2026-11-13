"""
Capítulo 02 - Leyendo la Caja de Cristal: Modelos Lineales
Entrena un regresor lineal y un clasificador logístico sobre features
previamente escaladas, y extrae sus coeficientes.
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from features import data

X = data.drop(columns=["approval_score", "approved"])
y_reg = data["approval_score"]
y_cls = data["approved"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

lin_reg = LinearRegression().fit(X_scaled, y_reg)
log_reg = LogisticRegression().fit(X_scaled, y_cls)

# Extrayendo los coeficientes
print(lin_reg.coef_)
print(log_reg.coef_)
