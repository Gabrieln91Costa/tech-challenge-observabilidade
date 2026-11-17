
import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "reference" / "credit_reference_clean.csv"

MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"

METRICS_PATH = (
    BASE_DIR / "reports" / "observability" / "baseline_metrics.json"
)

TARGET_COLUMN = "SeriousDlqin2yrs"

RANDOM_STATE = 42


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_data():
    print("\n========== CARREGANDO DADOS ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Registros: {len(df)}")
    print(f"Colunas: {len(df.columns)}")

    return df


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Coluna target não encontrada: {TARGET_COLUMN}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n========== DISTRIBUIÇÃO DO TARGET ==========")
    print(y.value_counts())

    print("\n========== DIVISÃO DOS DADOS ==========")
    print(f"Treinamento: {len(X_train)}")
    print(f"Teste: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# TREINAMENTO DO MODELO
# ============================================================

def train_model(X_train, y_train):

    print("\n========== TREINANDO RANDOM FOREST ==========")

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    print("Modelo treinado com sucesso!")

    return model


# ============================================================
# AVALIAÇÃO DO MODELO
# ============================================================

def evaluate_model(model, X_test, y_test):

    print("\n========== AVALIAÇÃO DO MODELO ==========")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RandomForestClassifier",
        "accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, probabilities), 4
        ),
        "test_records": int(len(y_test))
    }

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\n========== MATRIZ DE CONFUSÃO ==========")

    print(confusion_matrix(y_test, predictions))

    return metrics


# ============================================================
# SALVAMENTO DOS ARTEFATOS
# ============================================================

def save_artifacts(model, metrics):

    print("\n========== SALVANDO ARTEFATOS ==========")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Modelo: {MODEL_PATH}")
    print(f"Métricas: {METRICS_PATH}")


# ============================================================
# REGISTRO NO MLFLOW
# ============================================================

def log_mlflow(model, metrics):

    print("\n========== REGISTRANDO NO MLFLOW ==========")

    mlflow.set_experiment(
        "credit-scoring-observability"
    )

    with mlflow.start_run(
        run_name="random-forest-baseline"
    ):

        # Parâmetros do modelo
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        })

        # Métricas de avaliação
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        })

        # Registro do modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            name="credit-model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        # Registro do arquivo de métricas
        mlflow.log_artifact(
            str(METRICS_PATH)
        )

    print("Modelo e métricas registrados no MLflow!")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_artifacts(
        model,
        metrics
    )

    log_mlflow(
        model,
        metrics
    )

    print("\n========== PROCESSAMENTO FINALIZADO ==========")


if __name__ == "__main__":
    main()
