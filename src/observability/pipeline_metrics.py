import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from html import escape

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "reports" / "observability"
DATA_DRIFT_DIR = BASE_DIR / "reports" / "data_drift"
PRODUCTION_DIR = BASE_DIR / "data" / "production"
REFERENCE_DIR = BASE_DIR / "data" / "reference"
INVALID_DIR = BASE_DIR / "data" / "invalid"

OUTPUT_JSON = REPORT_DIR / "pipeline_metrics.json"
OUTPUT_HTML = REPORT_DIR / "operational_report.html"
OUTPUT_LOG = REPORT_DIR / "pipeline_execution.log"

BASELINE_PATH = REPORT_DIR / "baseline_metrics.json"
CONCEPT_PATH = REPORT_DIR / "concept_drift_evaluation.json"
DATA_DRIFT_PATH = DATA_DRIFT_DIR / "data_drift_metrics.json"

PRODUCTION_PATH = PRODUCTION_DIR / "credit_production.csv"
CONCEPT_DATA_PATH = PRODUCTION_DIR / "credit_concept_drift.csv"
REFERENCE_PATH = REFERENCE_DIR / "credit_reference_clean.csv"

# Limites experimentais de alerta
DRIFT_SHARE_THRESHOLD = 0.20
MIN_RECALL_DROP = -0.10


def setup_logging():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("credit_observability")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        OUTPUT_LOG,
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_json(path):

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def count_csv_records(path):

    if not path.exists():
        return None

    return int(len(pd.read_csv(path)))


def collect_data_quality():

    import hashlib

    clean_records = count_csv_records(REFERENCE_PATH)

    invalid_files = []
    unique_files = {}

    if INVALID_DIR.exists():

        for path in INVALID_DIR.glob("*.csv"):

            records = count_csv_records(path)

            # Calcula hash do conteúdo para identificar arquivos duplicados.
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            invalid_files.append({
                "file": path.name,
                "records": records
            })

            # Armazena apenas um arquivo por conteúdo.
            if file_hash not in unique_files:
                unique_files[file_hash] = records or 0

    # Soma somente os conteúdos únicos.
    rejected_records = sum(unique_files.values())

    return {
        "clean_reference_records": clean_records,
        "invalid_files": invalid_files,
        "rejected_records": rejected_records
    }



def collect_data_drift():

    data = load_json(DATA_DRIFT_PATH)

    if data is None:
        return {
            "available": False,
            "drift_detected": None,
            "drifted_columns": None,
            "total_columns": None,
            "drift_share": None,
            "threshold": DRIFT_SHARE_THRESHOLD,
            "column_metrics": []
        }

    metrics = data.get("metrics", [])

    dataset_metric = next(
        (
            item for item in metrics
            if item.get("metric_name", "").startswith(
                "DriftedColumnsCount"
            )
        ),
        None
    )

    if dataset_metric is None:
        raise ValueError(
            "Metrica DriftedColumnsCount nao encontrada no JSON."
        )

    value = dataset_metric.get("value", {})

    drifted_columns = int(value.get("count", 0))
    drift_share = float(value.get("share", 0))

    config = dataset_metric.get("config", {})
    total_columns = len(config.get("columns", []))

    column_metrics = []

    for item in metrics:

        if not item.get("metric_name", "").startswith("ValueDrift"):
            continue

        column_metrics.append({
            "metric": item.get("metric_name"),
            "value": item.get("value")
        })

    return {
        "available": True,
        "drift_detected": drift_share >= DRIFT_SHARE_THRESHOLD,
        "drifted_columns": drifted_columns,
        "total_columns": total_columns,
        "drift_share": drift_share,
        "threshold": DRIFT_SHARE_THRESHOLD,
        "column_metrics": column_metrics
    }


def collect_concept_drift():

    data = load_json(CONCEPT_PATH)

    if data is None:
        return {
            "available": False,
            "recall_delta": None,
            "roc_auc_delta": None,
            "original_metrics": {},
            "simulated_metrics": {}
        }

    original = data["original_metrics"]
    simulated = data["simulated_metrics"]
    deltas = data["metric_deltas"]

    return {
        "available": True,
        "model_retrained": data["model_retrained"],
        "records_evaluated": data["records_evaluated"],
        "recall_delta": deltas["recall"],
        "roc_auc_delta": deltas["roc_auc"],
        "original_metrics": original,
        "simulated_metrics": simulated,
        "false_negatives_before": original["confusion_matrix"][1][0],
        "false_negatives_after": simulated["confusion_matrix"][1][0]
    }


def collect_baseline():

    data = load_json(BASELINE_PATH)

    if data is None:
        return {"available": False}

    return {
        "available": True,
        "model": data["model"],
        "accuracy": data["accuracy"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1_score": data["f1_score"],
        "roc_auc": data["roc_auc"],
        "test_records": data["test_records"]
    }


def collect_artifact_status():

    artifacts = {
        "reference_data": REFERENCE_PATH,
        "production_data": PRODUCTION_PATH,
        "concept_drift_data": CONCEPT_DATA_PATH,
        "baseline_metrics": BASELINE_PATH,
        "data_drift_metrics": DATA_DRIFT_PATH,
        "concept_drift_evaluation": CONCEPT_PATH,
        "concept_drift_report": REPORT_DIR / "concept_drift_report.html"
    }

    return {
        name: {
            "available": path.exists(),
            "path": str(path.relative_to(BASE_DIR))
        }
        for name, path in artifacts.items()
    }


def generate_alerts(data_drift, concept_drift, artifacts):

    alerts = []

    if not data_drift["available"]:

        alerts.append({
            "severity": "WARNING",
            "message": "Relatorio de Data Drift indisponivel."
        })

    elif data_drift["drift_detected"]:

        alerts.append({
            "severity": "WARNING",
            "message": (
                f"Data Drift detectado em "
                f"{data_drift['drifted_columns']} de "
                f"{data_drift['total_columns']} variaveis."
            )
        })

    if not concept_drift["available"]:

        alerts.append({
            "severity": "WARNING",
            "message": "Avaliacao de Concept Drift indisponivel."
        })

    elif concept_drift["recall_delta"] <= MIN_RECALL_DROP:

        alerts.append({
            "severity": "CRITICAL",
            "message": (
                "Queda significativa de Recall no experimento "
                "de Concept Drift."
            )
        })

    if not all(
        item["available"] for item in artifacts.values()
    ):

        alerts.append({
            "severity": "WARNING",
            "message": "Existem artefatos esperados ausentes."
        })

    return alerts


def generate_html(report):

    data_drift = report["data_drift"]
    concept = report["concept_drift"]
    baseline = report["baseline"]
    volume = report["volume"]
    alerts = report["alerts"]

    alert_rows = ""

    if alerts:
        for alert in alerts:
            severity = escape(alert["severity"])
            message = escape(alert["message"])

            alert_rows += f"""
            <tr>
                <td class="{severity.lower()}">{severity}</td>
                <td>{message}</td>
            </tr>
            """
    else:
        alert_rows = """
        <tr><td colspan="2">Nenhum alerta registrado.</td></tr>
        """

    baseline_recall = (
        f"{baseline['recall']:.2%}"
        if baseline.get("available") else "N/D"
    )

    baseline_auc = (
        f"{baseline['roc_auc']:.2%}"
        if baseline.get("available") else "N/D"
    )

    drift_share = (
        f"{data_drift['drift_share']:.0%}"
        if data_drift["available"] else "N/D"
    )

    concept_recall = (
        f"{concept['simulated_metrics']['recall']:.2%}"
        if concept["available"] else "N/D"
    )

    production_records = volume["production_records"]
    production_label = (
        f"{production_records:,}"
        if production_records is not None else "N/D"
    )

    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Observabilidade Operacional - Credit Scoring</title>

<style>
body {{
    margin: 0;
    padding: 30px;
    font-family: Arial, sans-serif;
    background: #f3f5f8;
    color: #243044;
}}

.container {{
    max-width: 1100px;
    margin: auto;
}}

header {{
    background: #172b4d;
    color: white;
    padding: 30px;
    border-radius: 12px;
}}

section {{
    background: white;
    padding: 24px;
    margin-top: 22px;
    border-radius: 10px;
}}

.cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-top: 20px;
}}

.card {{
    background: #f1f4f8;
    padding: 20px;
    border-radius: 9px;
}}

.card span {{
    color: #64748b;
    font-size: 13px;
}}

.card strong {{
    display: block;
    font-size: 25px;
    margin-top: 10px;
}}

h2 {{
    color: #172b4d;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    padding: 12px;
    border-bottom: 1px solid #e5e9ef;
    text-align: left;
}}

th {{
    background: #f1f4f8;
}}

.critical {{
    color: #b42318;
    font-weight: bold;
}}

.warning {{
    color: #a15c00;
    font-weight: bold;
}}

.success {{
    color: #16834a;
}}

.note {{
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}}

footer {{
    text-align: center;
    color: #64748b;
    padding: 25px;
}}

@media(max-width: 700px) {{
    body {{ padding: 12px; }}
    .cards {{ grid-template-columns: 1fr; }}
}}
</style>
</head>

<body>
<div class="container">

<header>
    <h1>Observabilidade Operacional</h1>
    <p>Credit Scoring - Fintech</p>
    <p>Monitoramento de dados, modelo e alertas</p>
    <p>Gerado em: {escape(report["generated_at"])}</p>
</header>

<section>
    <h2>Indicadores principais</h2>

    <div class="cards">
        <div class="card">
            <span>Registros de produção</span>
            <strong>{production_label}</strong>
        </div>

        <div class="card">
            <span>Variáveis com Data Drift</span>
            <strong>{data_drift["drifted_columns"] if data_drift["available"] else "N/D"}/{data_drift["total_columns"] if data_drift["available"] else "N/D"}</strong>
        </div>

        <div class="card">
            <span>Share de Data Drift</span>
            <strong>{drift_share}</strong>
        </div>

        <div class="card">
            <span>Recall baseline</span>
            <strong>{baseline_recall}</strong>
        </div>

        <div class="card">
            <span>ROC-AUC baseline</span>
            <strong>{baseline_auc}</strong>
        </div>

        <div class="card">
            <span>Recall após Concept Drift</span>
            <strong>{concept_recall}</strong>
        </div>
    </div>
</section>

<section>
    <h2>Alertas operacionais</h2>

    <table>
        <thead>
            <tr>
                <th>Severidade</th>
                <th>Descrição</th>
            </tr>
        </thead>
        <tbody>
            {alert_rows}
        </tbody>
    </table>
</section>

<section>
    <h2>Qualidade dos dados</h2>

    <p>Registros limpos na referência:
       <strong>{report["data_quality"]["clean_reference_records"]}</strong>
    </p>

    <p>Registros rejeitados em arquivos de dados inválidos:
       <strong>{report["data_quality"]["rejected_records"]}</strong>
    </p>
</section>

<section>
    <h2>Execução e latência</h2>

    <p>Tempo de coleta e geração deste relatório:
       <strong>{report["execution"]["duration_seconds"]:.4f} segundos</strong>
    </p>

    <p class="note">
       Esse tempo representa somente a execução do coletor de métricas,
       não a latência histórica de todo o pipeline de treinamento e
       monitoramento.
    </p>
</section>

<section>
    <h2>Arquivos e evidências</h2>

    <table>
        <tr><th>Artefato</th><th>Disponível</th></tr>
        {
            "".join(
                f"<tr><td>{escape(name)}</td>"
                f"<td>{'Sim' if item['available'] else 'Não'}</td></tr>"
                for name, item in report["artifacts"].items()
            )
        }
    </table>
</section>

<footer>
    FIAP - Tech Challenge | Credit Scoring | Observabilidade
</footer>

</div>
</body>
</html>
"""

    OUTPUT_HTML.write_text(
        html_content,
        encoding="utf-8"
    )


def main():

    start = time.perf_counter()

    logger = setup_logging()

    logger.info("Inicio da coleta de observabilidade.")

    try:

        logger.info("Coletando metricas de baseline.")
        baseline = collect_baseline()

        logger.info("Coletando metricas de Data Drift.")
        data_drift = collect_data_drift()

        logger.info("Coletando metricas de Concept Drift.")
        concept_drift = collect_concept_drift()

        logger.info("Coletando indicadores de qualidade.")
        data_quality = collect_data_quality()

        logger.info("Verificando volume de registros.")

        volume = {
            "production_records": count_csv_records(PRODUCTION_PATH),
            "concept_drift_records": count_csv_records(CONCEPT_DATA_PATH),
            "reference_records": count_csv_records(REFERENCE_PATH)
        }

        logger.info("Verificando disponibilidade dos artefatos.")
        artifacts = collect_artifact_status()

        alerts = generate_alerts(
            data_drift,
            concept_drift,
            artifacts
        )

        duration = time.perf_counter() - start

        report = {
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "project": "credit-scoring",
            "execution": {
                "status": "success",
                "duration_seconds": round(duration, 4),
                "duration_scope": "metrics_collection_only"
            },
            "volume": volume,
            "data_quality": data_quality,
            "baseline": baseline,
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "artifacts": artifacts,
            "alerts": alerts,
            "alert_count": len(alerts)
        }

        with open(
            OUTPUT_JSON,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )

        generate_html(report)

        logger.info(
            "Coleta finalizada. Alertas identificados: %s",
            len(alerts)
        )

        logger.info("JSON: %s", OUTPUT_JSON)
        logger.info("HTML: %s", OUTPUT_HTML)
        logger.info("Log: %s", OUTPUT_LOG)

        print("\n==============================================")
        print(" OBSERVABILIDADE OPERACIONAL FINALIZADA")
        print("==============================================")
        print(f"Registros de producao: {volume['production_records']}")
        print(f"Variaveis com drift: {data_drift['drifted_columns']}")
        print(f"Alertas: {len(alerts)}")
        print(f"Tempo da coleta: {duration:.4f} segundos")

    except Exception:

        logger.exception("Falha na coleta de observabilidade.")
        raise


if __name__ == "__main__":
    main()
