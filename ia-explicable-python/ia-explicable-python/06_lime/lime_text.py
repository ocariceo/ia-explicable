"""
Capítulo 06 - LIME: Explicaciones Locales en Todas Partes
Texto.

Nota: 'sentiment_model' es un modelo de clasificación de sentimiento
con método predict_proba; no se define en el libro, debes aportar el tuyo.
"""
from lime.lime_text import LimeTextExplainer

review = "Fast approval, friendly staff, would recommend"

explainer = LimeTextExplainer(class_names=["negative", "positive"])
explanation = explainer.explain_instance(review, sentiment_model.predict_proba)
