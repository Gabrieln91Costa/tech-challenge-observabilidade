import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"

DATA_PATH = (
    BASE_DIR / "data" / "production" / "credit_concept_drift.csv"
)

REPORT_DIR = BASE_DIR / "reports" / "observability"

JSON_PATH = REPORT_DIR / "concept_drift_evaluation.json"

PREDICTIONS_PATH = (
    BASE_DIR / "data" / "production" / "concept_drift_predictions.csv"
)

TARGET_COLUMN = "SeriousDlqin2yrs"
ORIGINAL_TARGET = "target_original"


def calculate_metrics(y_true, predictions, probabilities):

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist()
    }


def main():

    print("\n==============================================")
    print(" AVALIACAO DE CONCEPT DRIFT - CREDIT SCORING")
    print("==============================================")

    # Carregar modelo congelado
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("\nModelo original carregado.")

    # Carregar dados simulados
    df = pd.read_csv(DATA_PATH)

    # Recuperar características utilizadas no treinamento
    X = df.drop(
        columns=[TARGET_COLUMN, ORIGINAL_TARGET]
    )

    y_original = df[ORIGINAL_TARGET]
    y_simulated = df[TARGET_COLUMN]

    # Gerar uma única previsão para cada registro
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Avaliar contra os dois conjuntos de rótulos
    original_metrics = calculate_metrics(
        y_original,
        predictions,
        probabilities
    )

    simulated_metrics = calculate_metrics(
        y_simulated,
        predictions,
        probabilities
    )

    print("\n========== BASELINE NA AMOSTRA ==========")

    for metric, value in original_metrics.items():
        print(f"{metric}: {value}")

    print("\n========== APOS CONCEPT DRIFT ==========")

    for metric, value in simulated_metrics.items():
        print(f"{metric}: {value}")

    # Calcular diferenças
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    deltas = {
        metric: round(
            simulated_metrics[metric] - original_metrics[metric],
            6
        )
        for metric in metric_names
    }

    # Salvar previsões
    df["prediction"] = predictions
    df["prediction_probability"] = probabilities

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PREDICTIONS_PATH, index=False)

    # Salvar relatório
    report = {
        "experiment": "concept_drift",
        "model_retrained": False,
        "records_evaluated": len(df),
        "original_metrics": original_metrics,
        "simulated_metrics": simulated_metrics,
        "metric_deltas": deltas
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== DIFERENCA DAS METRICAS ==========")

    for metric, delta in deltas.items():
        print(f"{metric}: {delta:+.6f}")

    print(f"\nRelatorio salvo: {JSON_PATH}")
    print(f"Previsoes salvas: {PREDICTIONS_PATH}")

    print("\nAVALIACAO FINALIZADA COM SUCESSO")


if __name__ == "__main__":
    main()
