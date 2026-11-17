# Tech Challenge — Observabilidade de Modelos de Machine Learning

## Credit Scoring para Fintech

**Projeto:** Monitoramento contínuo de dados e modelos de Machine Learning  
**Área:** Engenharia de Dados, Machine Learning e MLOps  
**Objetivo:** Implementar uma solução de observabilidade para identificar problemas de qualidade de dados, Data Drift e Concept Drift em um modelo de análise de risco de crédito.

---

## 1. Visão geral

Este projeto apresenta uma arquitetura experimental de observabilidade aplicada a um modelo de Credit Scoring de uma fintech.

O objetivo é demonstrar como o monitoramento contínuo pode identificar alterações na distribuição dos dados, degradação do desempenho preditivo e problemas de qualidade durante o ciclo de vida de um modelo de Machine Learning.

A solução contempla:

- Ingestão e preparação dos dados.
- Validação de contratos de dados.
- Identificação e tratamento de registros inválidos.
- Treinamento de modelo de classificação.
- Registro de métricas com MLflow.
- Simulação de Data Drift.
- Simulação controlada de Concept Drift.
- Monitoramento com Evidently AI.
- Geração de relatórios e alertas operacionais.
- Consolidação de indicadores de observabilidade.

## 2. Contexto do negócio

Instituições financeiras utilizam modelos de Credit Scoring para estimar o risco de inadimplência de clientes.

Entretanto, as características dos clientes e as condições econômicas podem mudar ao longo do tempo. Um modelo treinado com dados históricos pode perder capacidade preditiva quando essas mudanças não são monitoradas.

Nesse cenário, a ausência de observabilidade pode resultar em:

- Aumento de falsos negativos na identificação de clientes de risco.
- Redução da capacidade de identificar potenciais inadimplentes.
- Alterações inesperadas na distribuição dos dados.
- Decisões de crédito baseadas em previsões degradadas.
- Falta de rastreabilidade sobre a qualidade e o desempenho do modelo.

A solução proposta permite acompanhar esses indicadores e gerar evidências para apoiar decisões de manutenção e evolução do modelo.

## 3. Arquitetura da solução

```text
                    DATASET DE CRÉDITO
                            |
                            v
                   INGESTÃO DE DADOS
                            |
                            v
                 CONTRATO DE DADOS
                       PANDERA
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          DADOS VÁLIDOS          DADOS INVÁLIDOS
                 |                     |
                 v                     v
          PREPARAÇÃO E             REJEIÇÃO E
          TRATAMENTO              RASTREABILIDADE
                 |
                 v
          DATASET DE REFERÊNCIA
                 |
                 v
          TREINAMENTO DO MODELO
             RANDOM FOREST
                 |
                 v
             MODELO BASELINE
                 |
                 v
          REGISTRO DE MÉTRICAS
                MLFLOW
                 |
                 v
          SIMULAÇÃO DE PRODUÇÃO
                 |
          +------+------+
          |             |
          v             v
      DATA DRIFT   CONCEPT DRIFT
          |             |
          v             v
       EVIDENTLY    AVALIAÇÃO DO
          AI          MODELO
          |             |
          +------+------+
                 |
                 v
       OBSERVABILIDADE OPERACIONAL
                 |
          +------+------+
          |             |
          v             v
       MÉTRICAS       ALERTAS
          |             |
          v             v
       JSON / HTML / LOGS
```

### Principais componentes

| Componente | Responsabilidade |
|---|---|
| Pandas | Manipulação e preparação dos dados |
| Pandera | Validação dos contratos de dados |
| Scikit-learn | Treinamento e avaliação do modelo |
| MLflow | Registro de parâmetros, métricas e modelo |
| Evidently AI | Monitoramento de Data Drift |
| Python Logging | Registro das execuções |
| HTML e JSON | Relatórios e evidências de observabilidade |

## 4. Tecnologias utilizadas

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Pandera
- Evidently AI
- MLflow
- Matplotlib
- Git e GitHub

## 5. Dataset

Foi utilizado o dataset **Give Me Some Credit**, contendo informações relacionadas ao comportamento financeiro e histórico de crédito de clientes.

O conjunto original possui 150.000 registros.

A variável alvo é:

| Variável | Descrição |
|---|---|
| `SeriousDlqin2yrs` | Indica se ocorreu inadimplência grave nos próximos dois anos |

Distribuição original da variável alvo:

| Classe | Registros | Proporção |
|---|---:|---:|
| 0 — Sem inadimplência grave | 139.974 | 93,32% |
| 1 — Com inadimplência grave | 10.026 | 6,68% |

O dataset apresenta desbalanceamento entre as classes, característica importante para a avaliação do modelo.

### Preparação dos dados

Durante a preparação foram identificados:

- 609 registros duplicados excedentes.
- 14 registros com idade fora do intervalo permitido.
- Valores ausentes em `MonthlyIncome`.
- Valores ausentes em `NumberOfDependents`.

Os valores ausentes foram tratados utilizando imputação pela mediana.

Após o processo de preparação, o dataset de referência ficou com:

**149.377 registros válidos.**

Os registros rejeitados foram preservados em arquivos específicos para rastreabilidade.

## 6. Contrato de dados e qualidade

Foi implementado um contrato de dados utilizando Pandera.

O contrato estabelece regras de validação, incluindo:

- Tipo de dados esperado.
- Campos obrigatórios.
- Restrições de valores mínimos e máximos.
- Validação da variável alvo.
- Validação de idade entre 18 e 100 anos.
- Validação de valores não negativos em variáveis financeiras e quantitativas.
- Identificação de registros duplicados.

A ingestão é interrompida quando os dados não atendem às regras estabelecidas.

### Evidências de qualidade

| Indicador | Resultado |
|---|---:|
| Registros originais | 150.000 |
| Registros válidos | 149.377 |
| Duplicidades excedentes | 609 |
| Idades inválidas | 14 |
| Total de registros rejeitados | 623 |

Os arquivos de rejeição são mantidos no diretório `data/invalid/`.

## 7. Treinamento do modelo

Foi desenvolvido um classificador utilizando Random Forest.

### Configuração

| Parâmetro | Valor |
|---|---|
| Algoritmo | RandomForestClassifier |
| Número de estimadores | 150 |
| Profundidade máxima | 12 |
| Min samples split | 10 |
| Min samples leaf | 5 |
| Class weight | balanced |
| Random state | 42 |
| Test size | 20% |

A divisão entre treinamento e teste utiliza estratificação para preservar a proporção das classes.

### Métricas do modelo baseline

| Métrica | Resultado |
|---|---:|
| Accuracy | 82,42% |
| Precision | 23,87% |
| Recall | 74,13% |
| F1-score | 36,11% |
| ROC-AUC | 86,60% |

O modelo apresenta capacidade de discriminação, mas sua precisão para a classe positiva é limitada. Esse resultado deve ser considerado na avaliação de riscos e na definição de critérios de aprovação de crédito.

O modelo treinado é salvo em `models/credit_model.pkl`.

## 8. Registro de experimentos com MLflow

O MLflow foi utilizado para registrar o experimento de treinamento.

Foram registrados:

- Nome do experimento.
- Parâmetros do Random Forest.
- Métricas de avaliação.
- Artefato do modelo treinado.

Experimento:

`credit-scoring-observability`

Run:

`random-forest-baseline`

Essa estrutura permite rastrear as configurações e os resultados do treinamento baseline.

## 9. Monitoramento de Data Drift

Data Drift ocorre quando a distribuição dos dados de entrada sofre alterações em relação à distribuição utilizada como referência.

Para demonstrar esse comportamento, foi criada uma simulação de produção com 30.000 registros.

Foram aplicadas alterações controladas:

- Redução de 25% em `MonthlyIncome`.
- Aumento de 35% em `DebtRatio`.
- Aumento de 25% em `RevolvingUtilizationOfUnsecuredLines`.

O monitoramento foi realizado com Evidently AI, comparando os dados de referência com os dados simulados de produção.

### Resultado do experimento

| Variável | Wasserstein normalizado | Drift |
|---|---:|---|
| MonthlyIncome | 0,083949 | Detectado |
| DebtRatio | 0,076569 | Detectado |
| RevolvingUtilizationOfUnsecuredLines | 0,014316 | Não detectado |

Resultado consolidado:

- Variáveis monitoradas: 10.
- Variáveis com drift: 2.
- Proporção com drift: 20%.

Foi utilizada uma configuração experimental de `drift_share=0.20` e `num_threshold=0.05`.

Esses parâmetros foram definidos para a demonstração e não devem ser interpretados como limites universais para ambientes produtivos.

Relatório gerado:

`reports/data_drift/data_drift_report.html`

## 10. Monitoramento de Concept Drift

Concept Drift ocorre quando a relação entre as variáveis de entrada e a variável alvo se altera ao longo do tempo.

Diferentemente do Data Drift, essa alteração pode ocorrer mesmo quando a distribuição das variáveis de entrada permanece semelhante.

Para demonstrar esse cenário, foi realizado um experimento controlado:

1. Seleção de uma amostra de 30.000 registros.
2. Identificação de um segmento de clientes com maior DebtRatio e menor MonthlyIncome.
3. Alteração controlada de parte dos rótulos da classe negativa para a classe positiva.
4. Pres