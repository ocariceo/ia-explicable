"""
Capítulo 03 - Árboles que Muestran su Trabajo
Imprime las reglas de un árbol de decisión y calcula la importancia de
características de un random forest.
"""
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

from features import data

X = data.drop(columns=["approval_score", "approved"])
y = data["approved"]

# Un árbol ya es una explicación
tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
print(export_text(tree, feature_names=list(X.columns)))

# Los bosques aleatorios sacrifican claridad por precisión
forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
print(forest.feature_importances_)
