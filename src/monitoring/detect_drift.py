
import json
from pathlib import Path

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    BASE_DIR
    / "data"
    / "reference"
    / "credit_reference_clean.csv"
)

PRODUCTION_PATH = (
    BASE_DIR
    / "data"
    / "production"
    / "credit_production.csv"
)

REPORT_DIR = BASE_DIR / "reports" / "data_drift"

HTML_PATH = REPORT_DIR / "data_drift_report.html"

JSON_PATH = REPORT_DIR / "data_drift_metrics.json"

TARGET_COLUMN = "SeriousDlqin2yrs"

RANDOM_STATE = 43

DRIFT_SHARE_THRESHOLD = 0.20


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_data():

    print("\n========== CARREGANDO BASES ==========")

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Base de referencia nao encontrada: {REFERENCE_PATH}"
        )

    if not PRODUCTION_PATH.exists():
        raise FileNotFoundError(
            f"Base de producao nao encontrada: {PRODUCTION_PATH}"
        )

    reference_df = pd.read_csv(REFERENCE_PATH)

    production_df = pd.read_csv(PRODUCTION_PATH)

    print(f"Registros de referencia: {len(reference_df)}")
    print(f"Registros de producao: {len(production_df)}")

    return reference_df, production_df


# ============================================================
# PREPARACAO DAS BASES
# ============================================================

def prepare_data(reference_df, production_df):

    print("\n========== PREPARANDO DADOS PARA DRIFT ==========")

    # Selecionar somente as variaveis de entrada.
    # Target e previsoes nao entram na analise de Data Drift.

    feature_columns = [
        column
        for column in reference_df.columns
        if column != TARGET_COLUMN
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in production_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colunas ausentes na producao: {missing_columns}"
        )

    reference_features = reference_df[feature_columns].copy()

    production_features = production_df[feature_columns].copy()

    # Garantir tipos numericos.

    reference_features = reference_features.apply(
        pd.to_numeric,
        errors="raise"
    )

    production_features = production_features.apply(
        pd.to_numeric,
        errors="raise"
    )

    # Comparar amostras de mesmo tamanho para reduzir
    # o efeito de tamanhos diferentes nos testes estatisticos.

    sample_size = min(
        len(reference_features),
        len(production_features)
    )

    reference_features = reference_features.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    production_features = production_features.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"Variaveis analisadas: {len(feature_columns)}")
    print(f"Registros por base: {sample_size}")

    print("\nVariaveis monitoradas:")

    for column in feature_columns:
        print(f"- {column}")

    return reference_features, production_features, feature_columns


# ============================================================
# EXECUCAO DO DATA DRIFT
# ============================================================

def detect_drift(reference_features, production_features, feature_columns):
    report = Report([
        DataDriftPreset(
            columns=feature_columns,
            drift_share=DRIFT_SHARE_THRESHOLD,
            num_threshold=0.05
        )
    ])

    snapshot = report.run(
        current_data=production_features,
        reference_data=reference_features
    )

    return snapshot


# ============================================================
# SALVAMENTO DOS RELATORIOS
# ============================================================

def save_reports(snapshot):

    print("\n========== SALVANDO RELATORIOS ==========")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Relatorio visual HTML

    snapshot.save_html(
        str(HTML_PATH)
    )

    # Resultado completo da avaliacao em JSON

    snapshot.save_json(
        str(JSON_PATH)
    )

    print(f"Relatorio HTML: {HTML_PATH}")
    print(f"Metricas JSON: {JSON_PATH}")


# ============================================================
# RESUMO DA EXECUCAO
# ============================================================

def save_execution_summary(reference_features, production_features):

    summary = {
        "monitoring": "Data Drift",
        "reference_records": int(len(reference_features)),
        "production_records": int(len(production_features)),
        "features_analyzed": list(reference_features.columns),
        "features_count": int(len(reference_features.columns)),
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "reference_dataset": str(REFERENCE_PATH),
        "production_dataset": str(PRODUCTION_PATH)
    }

    summary_path = REPORT_DIR / "execution_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Resumo da execucao: {summary_path}")


# ============================================================
# EXECUCAO PRINCIPAL
# ============================================================

def main():

    print("\n==============================================")
    print(" MONITORAMENTO DE DATA DRIFT - CREDIT SCORING")
    print("==============================================")

    reference_df, production_df = load_data()

    reference_features, production_features, feature_columns = (
        prepare_data(
            reference_df,
            production_df
        )
    )

    snapshot = detect_drift(
        reference_features,
        production_features,
        feature_columns
    )

    save_reports(snapshot)

    save_execution_summary(
        reference_features,
        production_features
    )

    print("\n==============================================")
    print(" MONITORAMENTO FINALIZADO COM SUCESSO")
    print("==============================================")


if __name__ == "__main__":
    main()
