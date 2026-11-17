import json
import html
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR / "reports" / "observability"
    / "concept_drift_evaluation.json"
)

OUTPUT_PATH = (
    BASE_DIR / "reports" / "observability"
    / "concept_drift_report.html"
)


METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "roc_auc": "ROC-AUC"
}


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_delta(value):
    return f"{value * 100:+.2f} p.p."


def metric_status(delta):
    if delta < -0.05:
        return "negative"
    if delta > 0.05:
        return "positive"
    return "neutral"


def build_metric_rows(report):

    rows = ""

    original = report["original_metrics"]
    simulated = report["simulated_metrics"]
    deltas = report["metric_deltas"]

    for key, label in METRICS.items():

        before = original[key]
        after = simulated[key]
        delta = deltas[key]

        status = metric_status(delta)

        rows += f"""
        <tr>
            <td>{label}</td>
            <td>{format_percent(before)}</td>
            <td>{format_percent(after)}</td>
            <td class="{status}">{format_delta(delta)}</td>
        </tr>
        """

    return rows


def build_confusion_matrix(matrix):

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    return f"""
    <table class="matrix">
        <tr>
            <th></th>
            <th>Previsto: 0</th>
            <th>Previsto: 1</th>
        </tr>
        <tr>
            <th>Real: 0</th>
            <td class="good">{tn:,}</td>
            <td>{fp:,}</td>
        </tr>
        <tr>
            <th>Real: 1</th>
            <td class="bad">{fn:,}</td>
            <td class="good">{tp:,}</td>
        </tr>
    </table>
    """


def generate_report():

    print("\n========== GERANDO RELATORIO CONCEPT DRIFT ==========")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Relatorio de avaliacao nao encontrado: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    original = report["original_metrics"]
    simulated = report["simulated_metrics"]

    rows = build_metric_rows(report)

    original_matrix = build_confusion_matrix(
        original["confusion_matrix"]
    )

    simulated_matrix = build_confusion_matrix(
        simulated["confusion_matrix"]
    )

    recall_delta = report["metric_deltas"]["recall"]
    auc_delta = report["metric_deltas"]["roc_auc"]

    false_negative_before = original["confusion_matrix"][1][0]
    false_negative_after = simulated["confusion_matrix"][1][0]

    fn_increase = false_negative_after - false_negative_before

    total_records = report["records_evaluated"]

    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Relatório de Concept Drift - Credit Scoring</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px;
    font-family: Arial, sans-serif;
    background: #f3f5f8;
    color: #243044;
}}

.container {{
    max-width: 1150px;
    margin: auto;
}}

header {{
    background: #172b4d;
    color: white;
    padding: 35px;
    border-radius: 12px;
    margin-bottom: 25px;
}}

header h1 {{
    margin: 0 0 12px;
    font-size: 28px;
}}

header p {{
    color: #dce5f2;
    margin: 5px 0;
}}

section {{
    background: white;
    padding: 25px;
    margin-bottom: 22px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}}

h2 {{
    margin-top: 0;
    color: #172b4d;
}}

.cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}}

.card {{
    padding: 20px;
    background: #f4f6fa;
    border-radius: 9px;
}}

.card span {{
    color: #64748b;
    font-size: 13px;
}}

.card strong {{
    display: block;
    font-size: 27px;
    margin-top: 10px;
}}

.danger {{
    color: #c0392b;
}}

.success {{
    color: #16834a;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}}

th, td {{
    padding: 13px;
    border-bottom: 1px solid #e5e9ef;
    text-align: center;
}}

th {{
    background: #f0f3f8;
}}

td:first-child {{
    text-align: left;
    font-weight: bold;
}}

.negative {{
    color: #c0392b;
    font-weight: bold;
}}

.positive {{
    color: #16834a;
    font-weight: bold;
}}

.neutral {{
    color: #64748b;
}}

.matrices {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
}}

.matrix td {{
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}}

.matrix th {{
    font-size: 12px;
}}

.good {{
    background: #e4f4e9;
    color: #16713b;
}}

.bad {{
    background: #fde8e7;
    color: #b42318;
}}

.highlight {{
    border-left: 5px solid #c0392b;
    background: #fff5f4;
    padding: 18px;
    line-height: 1.6;
}}

.note {{
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}}

footer {{
    text-align: center;
    padding: 20px;
    color: #64748b;
    font-size: 12px;
}}

@media(max-width: 750px) {{
    body {{
        padding: 12px;
    }}

    .cards, .matrices {{
        grid-template-columns: 1fr;
    }}

    table {{
        font-size: 12px;
    }}

    th, td {{
        padding: 8px 4px;
    }}
}}
</style>
</head>

<body>
<div class="container">

<header>
    <h1>Monitoramento de Concept Drift</h1>
    <p>Projeto: Credit Scoring - Fintech</p>
    <p>Experimento de mudança na relação entre características e inadimplência</p>
    <p>Modelo avaliado: Random Forest congelado, sem retreinamento</p>
</header>

<section>
    <h2>1. Resumo executivo</h2>

    <div class="cards">
        <div class="card">
            <span>Registros avaliados</span>
            <strong>{total_records:,}</strong>
        </div>

        <div class="card">
            <span>Queda no Recall</span>
            <strong class="danger">{format_delta(recall_delta)}</strong>
        </div>

        <div class="card">
            <span>Queda no ROC-AUC</span>
            <strong class="danger">{format_delta(auc_delta)}</strong>
        </div>
    </div>

    <div class="highlight" style="margin-top:20px">
        <strong>Resultado:</strong>
        a alteração simulada nos rótulos provocou uma redução
        expressiva na capacidade do modelo de identificar clientes
        inadimplentes.
    </div>
</section>

<section>
    <h2>2. Comparação das métricas</h2>

    <p class="note">
        Comparação do mesmo modelo e das mesmas previsões,
        avaliados contra os rótulos originais e os rótulos
        modificados na simulação.
    </p>

    <table>
        <thead>
            <tr>
                <th>Métrica</th>
                <th>Antes</th>
                <th>Após</th>
                <th>Variação</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</section>

<section>
    <h2>3. Matrizes de confusão</h2>

    <div class="matrices">
        <div>
            <h3>Rótulos originais</h3>
            {original_matrix}
        </div>

        <div>
            <h3>Rótulos simulados</h3>
            {simulated_matrix}
        </div>
    </div>

    <p class="note">
        Verde: classificações corretas.
        Vermelho: falsos negativos, ou seja, inadimplentes
        classificados como não inadimplentes.
    </p>
</section>

<section>
    <h2>4. Impacto no risco de crédito</h2>

    <div class="highlight">
        Os falsos negativos aumentaram de
        <strong>{false_negative_before:,}</strong> para
        <strong>{false_negative_after:,}</strong> registros.

        Isso representa um aumento de
        <strong>{fn_increase:,}</strong> casos não identificados
        pelo modelo na simulação.
    </div>

    <p>
        Em um cenário de concessão de crédito, essa perda de
        sensibilidade pode elevar o risco de aprovar operações
        para clientes que apresentam inadimplência.
    </p>
</section>

<section>
    <h2>5. Metodologia do experimento</h2>

    <ul>
        <li>Utilização de 30.000 registros da base de referência.</li>
        <li>Seleção de um segmento com DebtRatio elevado e renda mensal baixa.</li>
        <li>Alteração artificial de parte dos rótulos de inadimplência.</li>
        <li>Preservação das características originais dos registros.</li>
        <li>Manutenção do modelo treinado, sem retreinamento.</li>
        <li>Comparação das métricas contra os dois conjuntos de rótulos.</li>
    </ul>

    <p class="note">
        Este é um experimento sintético e controlado para demonstrar
        mudança em P(Y|X). Não representa evidência de mudança real
        no comportamento dos clientes ou no mercado de crédito.
        A amostra também é utilizada para comparação experimental,
        não constituindo uma validação independente do modelo.
    </p>
</section>

<section>
    <h2>6. Conclusão e ações recomendadas</h2>

    <p>
        O experimento demonstrou que a alteração da relação entre
        características financeiras e inadimplência pode comprometer
        o desempenho de um modelo de Credit Scoring.
    </p>

    <p>
        Recomenda-se monitorar continuamente Recall, F1-score,
        ROC-AUC e falsos negativos, investigar mudanças no
        comportamento dos dados rotulados e avaliar a necessidade
        de retreinamento após validação e análise de governança.
    </p>
</section>

<footer>
    Relatório gerado automaticamente pelo pipeline de observabilidade.
    <br>
    FIAP - Tech Challenge - Credit Scoring
</footer>

</div>
</body>
</html>
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        html_content,
        encoding="utf-8"
    )

    print("\nRelatorio HTML gerado com sucesso!")
    print(f"Arquivo: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_report()
