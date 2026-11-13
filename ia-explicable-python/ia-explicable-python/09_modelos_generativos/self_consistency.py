"""
Capítulo 09 - Explicando Modelos Generativos y de Chat
Autocoherencia (Self-consistency): ¿qué tan seguro está el modelo?

Nota: 'get_response' envuelve tu llamada a la API del modelo.
"""
from collections import Counter

prompt = (
    "Classify the sentiment of this review as exactly one word, "
    "'positive' or 'negative': 'Approval took forever and the "
    "rate was worse than advertised.'"
)

responses = [get_response(prompt).strip().lower() for _ in range(5)]

counts = Counter(responses)
confidence = {label: n / len(responses) for label, n in counts.items()}
print(confidence)

# OUT esperado:
# {'negative': 0.8, 'positive': 0.2}
