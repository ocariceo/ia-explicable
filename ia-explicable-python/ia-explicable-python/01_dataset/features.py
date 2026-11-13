"""
Conjunto de datos sintético de aprobación de hipotecas, usado en todos los
capítulos de la guía "IA Explicable con Python".
"""
import pandas as pd

data = pd.DataFrame({
    "annual_income_k": [82, 41, 63, 120, 35, 97, 54, 30],
    "credit_score": [710, 590, 655, 780, 560, 705, 615, 540],
    "debt_to_income": [0.21, 0.44, 0.33, 0.15, 0.52, 0.19, 0.38, 0.61],
    "years_employed": [6, 1, 3, 11, 0, 8, 2, 1],
    "loan_amount_k": [220, 180, 150, 300, 140, 260, 175, 160],
    "approval_score": [0.88, 0.31, 0.62, 0.95, 0.20, 0.90, 0.48, 0.12],
    "approved": [1, 0, 1, 1, 0, 1, 0, 0],
})
