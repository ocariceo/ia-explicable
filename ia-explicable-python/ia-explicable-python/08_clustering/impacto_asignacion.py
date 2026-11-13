"""
Capítulo 08 - Explicando Modelos Sin Etiquetas
Impacto en la asignación: Índice de Rand ajustado (ARI).
"""
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

from features import data

X = data.drop(columns=["approval_score", "approved"])
X_arr = X.values

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_arr)
original_clusters = kmeans.predict(X_arr)

assignment_impact = {}
for i, feature in enumerate(X.columns):
    reduced = np.delete(X_arr, i, axis=1)
    reduced_clusters = KMeans(n_clusters=2, random_state=0,
                               n_init=10).fit_predict(reduced)
    assignment_impact[feature] = 1 - adjusted_rand_score(original_clusters,
                                                           reduced_clusters)

print(assignment_impact)
