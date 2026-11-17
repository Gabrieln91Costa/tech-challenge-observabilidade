
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    BASE_DIR / "data" / "reference" / "credit_reference_clean.csv"
)

MODEL_PATH = (
    BASE_DIR / "models" / "credit_model.pkl"
)

PRODUCTION_PATH = (
    BASE_DIR / "data" / "production" / "credit_production.csv"
)

TARGET_COLUMN = "SeriousDlqin2yrs"

PRODUCTION_SIZE = 30000

RANDOM_STATE = 42


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_reference_data():

    print("\n========== CARREGANDO BASE DE REFERÊNCIA ==========")

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Base de referência não encontrada: {REFERENCE_PATH}"
        )

    df = pd.read_csv(REFERENCE_PATH)

    print(f"Registros disponíveis: {len(df)}")

    return df


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

def load_model():

    print("\n========== CARREGANDO MODELO TREINADO ==========")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Modelo carregado com sucesso!")

    return model


# ============================================================
# SIMULAÇÃO DE DATA DRIFT
# ============================================================

def simulate_production_data(df_reference):

    print("\n========== SIMULANDO DADOS DE PRODUÇÃO ==========")

    production_size = min(
        PRODUCTION_SIZE,
        len(df_reference)
    )

    df_production = df_reference.sample(
        n=production_size,
        random_state=RANDOM_STATE
    ).copy()

    # --------------------------------------------------------
    # DRIFT 1: REDUÇÃO DA RENDA MENSAL
    # --------------------------------------------------------

    df_production["MonthlyIncome"] *= 0.75

    # --------------------------------------------------------
    # DRIFT 2: AUMENTO DO COMPROMETIMENTO DE RENDA
    # --------------------------------------------------------

    df_production["DebtRatio"] *= 1.35

    # --------------------------------------------------------
    # DRIFT 3: AUMENTO DA UTILIZAÇÃO DO CRÉDITO
    # --------------------------------------------------------

    df_production[
        "RevolvingUtilizationOfUnsecuredLines"
    ] *= 1.25

    print(f"Registros de produção: {len(df_production)}")

    print("\nAlterações simuladas:")

    print("MonthlyIncome: redução de 25%")
    print("DebtRatio: aumento de 35%")
    print("RevolvingUtilizationOfUnsecuredLines: aumento de 25%")

    return df_production


# ============================================================
# GERAÇÃO DE PREVISÕES
# ============================================================

def generate_predictions(model, df_production):

    print("\n========== GERANDO PREVISÕES ==========")

    X_production = df_production.drop(
        columns=[TARGET_COLUMN]
    )

    predictions = model.predict(X_production)

    probabilities = model.predict_proba(
        X_production
    )[:, 1]

    df_production["prediction"] = predictions

    df_production["prediction_probability"] = probabilities

    print("Previsões geradas com sucesso!")

    print("\nDistribuição das previsões:")

    print(
        df_production["prediction"].value_counts()
    )

    return df_production


# ============================================================
# SALVAMENTO DOS DADOS
# ============================================================

def save_production_data(df_production):

    print("\n========== SALVANDO DADOS DE PRODUÇÃO ==========")

    PRODUCTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_production.to_csv(
        PRODUCTION_PATH,
        index=False
    )

    print(f"Arquivo salvo: {PRODUCTION_PATH}")

    print(f"Registros: {len(df_production)}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    df_reference = load_reference_data()

    model = load_model()

    df_production = simulate_production_data(
        df_reference
    )

    df_production = generate_predictions(
        model,
        df_production
    )

    save_production_data(
        df_production
    )

    print("\n========== SIMULAÇÃO FINALIZADA ==========")


if __name__ == "__main__":
    main()
