"""
Capítulo 09 - Explicando Modelos Generativos y de Chat
Prompting de Cadena de Pensamiento (Chain-of-thought).

Nota: 'get_response' envuelve tu llamada a la API del modelo; no se define
en el libro, debes aportar la tuya (p. ej. usando la API de Anthropic/OpenAI).
"""
prompt = (
    "A warehouse starts with 84 units of stock. It ships out "
    "27 units, receives a delivery of 40 units, then ships out "
    "15 more. Think through this step by step, then give the final count."
)

response = get_response(prompt)  # envuelve tu llamada a la API del modelo
print(response)

# OUT esperado:
# Starting stock: 84
# After shipping 27: 84 - 27 = 57
# After receiving 40: 57 + 40 = 97
# After shipping 15: 97 - 15 = 82
# Final count: 82 units
