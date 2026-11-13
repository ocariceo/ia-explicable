"""
Capítulo 07 - ¿Es Buena la Explicación?
Fidelidad: ¿la explicación predice el comportamiento del propio modelo?

Nota: requiere 'log_reg' y 'scaler' ya entrenados (ver 02_modelos_lineales).
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from features import data

X = data.drop(columns=["approval_score", "approved"])
y_cls = data["approved"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
log_reg = LogisticRegression().fit(X_scaled, y_cls)

instance = X.iloc[[1]]
original_prob = log_reg.predict_proba(scaler.transform(instance))[0][1]

# LIME marcó credit_score como algo que bajaba esta predicción; súbela y
# vuelve a revisar
bumped = instance.copy()
bumped["credit_score"] += 60
new_prob = log_reg.predict_proba(scaler.transform(bumped))[0][1]

faithfulness = abs(new_prob - original_prob)
print(original_prob, new_prob, faithfulness)  # Salida: 0.31 0.68 0.37
