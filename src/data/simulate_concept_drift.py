import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    BASE_DIR / "data" / "reference" / "credit_reference_clean.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "production" / "credit_concept_drift.csv"
)

SUMMARY_PATH = (
    BASE_DIR / "reports" / "observability" / "concept_drift_simulation.json"
)

TARGET_COLUMN = "SeriousDlqin2yrs"

PRODUCTION_SIZE = 30000
RANDOM_STATE = 42
LABEL_CHANGE_RATE = 0.50


def main():

    print("\n==============================================")
    print(" SIMULACAO DE CONCEPT DRIFT - CREDIT SCORING")
    print("==============================================")

    # 1. Carregar referência
    df_reference = pd.read_csv(REFERENCE_PATH)

    print(f"\nRegistros de referencia: {len(df_reference)}")

    # 2. Selecionar amostra
    df_production = df_reference.sample(
        n=min(PRODUCTION_SIZE, len(df_reference)),
        random_state=RANDOM_STATE
    ).copy().reset_index(drop=True)

    # 3. Preservar rótulos originais
    original_target = df_production[TARGET_COLUMN].copy()

    # 4. Definir segmento de risco
    debt_threshold = df_production["DebtRatio"].quantile(0.75)
    income_threshold = df_production["MonthlyIncome"].median()

    risk_segment = (
        (df_production["DebtRatio"] >= debt_threshold)
        & (df_production["MonthlyIncome"] <= income_threshold)
    )

    # 5. Alterar a relação entre características e inadimplência
    rng = np.random.default_rng(RANDOM_STATE)

    eligible_indices = df_production.index[
        risk_segment & (original_target == 0)
    ].to_numpy()

    number_to_change = int(
        len(eligible_indices) * LABEL_CHANGE_RATE
    )

    selected_indices = rng.choice(
        eligible_indices,
        size=number_to_change,
        replace=False
    )

    df_production["target_original"] = original_target

    df_production.loc[
        selected_indices,
        TARGET_COLUMN
    ] = 1

    # 6. Estatísticas
    original_rate = float(original_target.mean())
    new_rate = float(df_production[TARGET_COLUMN].mean())

    changed_labels = int(
        (df_production[TARGET_COLUMN] != original_target).sum()
    )

    print("\n========== RESULTADO DA SIMULACAO ==========")

    print(f"Registros de producao: {len(df_production)}")
    print(f"Clientes no segmento de risco: {int(risk_segment.sum())}")
    print(f"Rotulos alterados: {changed_labels}")

    print(f"\nInadimplencia original: {original_rate:.2%}")
    print(f"Inadimplencia simulada: {new_rate:.2%}")

    print(f"\nLimite DebtRatio: {debt_threshold:.4f}")
    print(f"Limite MonthlyIncome: {income_threshold:.2f}")

    # 7. Salvar dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_production.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 8. Salvar resumo
    summary = {
        "experiment": "concept_drift",
        "records": len(df_production),
        "risk_segment_records": int(risk_segment.sum()),
        "changed_labels": changed_labels,
        "label_change_rate": LABEL_CHANGE_RATE,
        "original_default_rate": original_rate,
        "simulated_default_rate": new_rate,
        "debt_ratio_threshold": float(debt_threshold),
        "monthly_income_threshold": float(income_threshold),
        "random_state": RANDOM_STATE,
        "feature_distribution_preserved": True
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print(f"\nDataset salvo: {OUTPUT_PATH}")
    print(f"Resumo salvo: {SUMMARY_PATH}")

    print("\n==============================================")
    print(" SIMULACAO DE CONCEPT DRIFT FINALIZADA")
    print("==============================================")


if __name__ == "__main__":
    main()
