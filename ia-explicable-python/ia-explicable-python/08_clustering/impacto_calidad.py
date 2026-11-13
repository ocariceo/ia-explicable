"""
Capítulo 08 - Explicando Modelos Sin Etiquetas
Impacto en la calidad del cluster: leave-one-out con puntaje de silueta.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

from features import data

X = data.drop(columns=["approval_score", "approved"])
X_arr = X.values

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_arr)
original_score = silhouette_score(X_arr, kmeans.labels_)

impact = {}
for i, feature in enumerate(X.columns):
    reduced = np.delete(X_arr, i, axis=1)
    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(reduced)
    new_score = silhouette_score(reduced, labels)
    impact[feature] = original_score - new_score

print(impact)
