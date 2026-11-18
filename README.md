# Tech Challenge — Observabilidade de Modelos de Machine Learning

## Credit Scoring para Fintech

**Projeto:** Monitoramento contínuo de dados e modelos de Machine Learning  
**Área:** Engenharia de Dados, Machine Learning e MLOps  
**Objetivo:** Implementar uma solução experimental de observabilidade para identificar problemas de qualidade de dados, Data Drift e Concept Drift em um modelo de análise de risco de crédito.

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
- Monitoramento estatístico com Evidently AI.
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
          PREPARAÇÃO E              REJEIÇÃO E
           TRATAMENTO             RASTREABILIDADE
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
          +------+------+
                 |
                 v
             JSON / HTML / LOGS
```

### Principais componentes

| Componente | Responsabilidade |
|---|---|
| Pandas | Manipulação e preparação dos dados |
| Pandera | Validação dos contratos de dados |
| Scikit-learn | Treinamento e avaliação do modelo |
| MLflow | Registro de parâmetros, métricas e modelo |
| Evidently AI | Monitoramento estatístico de Data Drift |
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

**Experimento:** `credit-scoring-observability`  
**Run:** `random-forest-baseline`

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

**Relatório gerado:** `reports/data_drift/data_drift_report.html`

## 10. Monitoramento de Concept Drift

Concept Drift ocorre quando a relação entre as variáveis de entrada e a variável alvo se altera ao longo do tempo.

Diferentemente do Data Drift, essa alteração pode ocorrer mesmo quando a distribuição das variáveis de entrada permanece semelhante.

Para demonstrar esse cenário, foi realizado um experimento controlado:

1. Seleção de uma amostra de 30.000 registros.
2. Identificação de um segmento de clientes com maior `DebtRatio` e menor `MonthlyIncome`.
3. Alteração controlada de parte dos rótulos da classe negativa para a classe positiva.
4. Preservação das distribuições das variáveis de entrada.
5. Avaliação do modelo baseline sem realizar novo treinamento.

A simulação alterou 3.402 rótulos de inadimplência dentro do segmento selecionado.

A taxa de inadimplência da amostra passou de aproximadamente 6,54% para 17,88%.

**Importante:** trata-se de uma simulação controlada para demonstrar a degradação do modelo. Não representa uma mudança real observada no mercado financeiro.

### Comparação de desempenho

| Métrica | Cenário original | Cenário simulado |
|---|---:|---:|
| Accuracy | 83,06% | 75,21% |
| Precision | 25,15% | 33,50% |
| Recall | 80,48% | 39,21% |
| F1-score | 38,33% | 36,13% |
| ROC-AUC | 90,18% | 59,95% |

### Variação das métricas

| Métrica | Variação |
|---|---:|
| Accuracy | -7,85 p.p. |
| Precision | +8,35 p.p. |
| Recall | -41,27 p.p. |
| F1-score | -2,20 p.p. |
| ROC-AUC | -30,23 p.p. |

O principal alerta observado foi a queda do Recall, indicando uma redução expressiva da capacidade do modelo de identificar corretamente os casos positivos.

### Impacto nos falsos negativos

| Indicador | Cenário original | Cenário simulado |
|---|---:|---:|
| Falsos negativos | 383 | 3.261 |

O aumento de falsos negativos demonstra o risco de manter um modelo sem monitoramento adequado quando o comportamento da variável alvo muda.

**Relatório gerado:** `reports/observability/concept_drift_report.html`

**Resultado da avaliação:** `reports/observability/concept_drift_evaluation.json`

## 11. Observabilidade operacional

Foi desenvolvido um módulo de observabilidade para consolidar indicadores de qualidade, Data Drift, Concept Drift e execução dos componentes do projeto.

O módulo centraliza informações provenientes dos relatórios JSON, arquivos CSV e logs gerados durante os experimentos.

### Indicadores monitorados

| Categoria | Indicador |
|---|---|
| Qualidade de dados | Registros válidos |
| Qualidade de dados | Registros rejeitados |
| Qualidade de dados | Volume de dados de referência |
| Data Drift | Quantidade de variáveis com drift |
| Data Drift | Proporção de variáveis com drift |
| Concept Drift | Variação do Recall |
| Concept Drift | Variação do ROC-AUC |
| Execução | Tempo de coleta das métricas |
| Integridade | Disponibilidade dos artefatos esperados |

### Regras de alerta

Foram implementadas regras experimentais para demonstrar a identificação de situações críticas:

| Condição | Severidade | Interpretação |
|---|---|---|
| Proporção de variáveis com drift igual ou superior a 20% | WARNING | Alteração relevante na distribuição dos dados |
| Queda do Recall igual ou superior a 10 p.p. | CRITICAL | Possível degradação relevante do modelo |
| Ausência de artefatos esperados | WARNING | Problema de integridade ou execução |

### Resultado operacional consolidado

- Dataset de referência: 149.377 registros.
- Dataset de produção simulado: 30.000 registros.
- Registros rejeitados: 623.
- Variáveis com Data Drift: 2 de 10.
- Recall baseline: 74,13%.
- Recall após Concept Drift: 39,21%.
- ROC-AUC baseline: 86,60%.
- ROC-AUC após Concept Drift: 59,95%.
- Alertas gerados: 2.

### Artefatos gerados

| Arquivo | Finalidade |
|---|---|
| `pipeline_metrics.json` | Indicadores consolidados |
| `operational_report.html` | Relatório operacional |
| `pipeline_execution.log` | Histórico de execução |

Os artefatos são armazenados em `reports/observability/`.

**Observação:** os alertas apresentados são gerados a partir dos resultados dos experimentos e dos artefatos disponíveis. Esta entrega não implementa monitoramento online contínuo nem integração com uma plataforma externa de alertas.

## 12. Estrutura do projeto

```text
tech-challenge-observabilidade/
│
├── data/
│   ├── reference/
│   ├── production/
│   └── invalid/
│
├── models/
│
├── reports/
│   ├── data_quality/
│   ├── data_drift/
│   └── observability/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── simulate_drift.py
│   │   └── simulate_concept_drift.py
│   │
│   ├── validation/
│   │   └── data_contract.py
│   │
│   ├── training/
│   │   └── train_model.py
│   │
│   ├── monitoring/
│   │   ├── detect_drift.py
│   │   ├── evaluate_concept_drift.py
│   │   └── generate_concept_drift_report.py
│   │
│   └── observability/
│       └── pipeline_metrics.py
│
├── tests/
│   └── test_data_contract.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 13. Como executar o projeto

### 13.1. Pré-requisitos

- Python 3.11.
- Git.
- Windows, Linux ou macOS.
- Ambiente virtual Python.

### 13.2. Clonar o repositório

```bash
git clone https://github.com/Gabrieln91Costa/tech-challenge-observabilidade.git

cd tech-challenge-observabilidade
```

### 13.3. Criar e ativar o ambiente virtual

No Windows PowerShell:

```powershell
uv venv --python 3.11 .venv

.\.venv\Scripts\Activate.ps1
```

### 13.4. Instalar dependências

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

### 13.5. Executar o pipeline

Execute os scripts na sequência abaixo, respeitando a geração dos artefatos utilizados pelas etapas seguintes:

```powershell
python -m src.data.load_data

python -m src.training.train_model

python -m src.data.simulate_drift

python -m src.monitoring.detect_drift

python -m src.data.simulate_concept_drift

python -m src.monitoring.evaluate_concept_drift

python -m src.monitoring.generate_concept_drift_report

python -m src.observability.pipeline_metrics
```

Os comandos devem ser executados a partir da raiz do repositório.

### 13.6. Executar os testes

```powershell
python -m pytest -v
```

Para verificar a compilação dos módulos:

```powershell
python -m compileall -q src tests
```

### 13.7. Visualizar os relatórios

Após a execução, os relatórios HTML estarão disponíveis nos diretórios:

```text
reports/data_drift/data_drift_report.html

reports/observability/concept_drift_report.html

reports/observability/operational_report.html
```

O MLflow também pode ser iniciado localmente, quando necessário:

```powershell
mlflow ui
```

A interface padrão estará disponível em `http://localhost:5000`.

## 14. Governança de Dados, Privacidade e LGPD

### 14.1. Objetivo da governança

A governança de dados tem como objetivo estabelecer diretrizes para o uso responsável, seguro e rastreável das informações utilizadas no modelo de Credit Scoring.

Em uma aplicação financeira real, o tratamento de dados deve observar a Lei Geral de Proteção de Dados Pessoais (LGPD — Lei nº 13.709/2018), considerando princípios como:

- Finalidade.
- Adequação.
- Necessidade.
- Livre acesso.
- Qualidade dos dados.
- Transparência.
- Segurança.
- Prevenção.
- Não discriminação.
- Responsabilização e prestação de contas.

A governança deve acompanhar todo o ciclo de vida dos dados, desde a coleta até o descarte.

### 14.2. Classificação e tratamento dos dados

O projeto utiliza o dataset público Give Me Some Credit, composto por informações relacionadas ao histórico e comportamento financeiro.

Entre os atributos utilizados estão:

- Idade.
- Renda mensal.
- Relação entre dívidas e renda.
- Utilização de crédito rotativo.
- Quantidade de linhas de crédito.
- Histórico de atrasos.
- Quantidade de dependentes.

Essas informações podem ser consideradas dados pessoais quando relacionadas a uma pessoa identificada ou identificável.

A ausência de campos como nome, CPF, telefone e endereço no conjunto de atributos do modelo não significa, por si só, que os dados estejam anonimizados.

Também é importante observar que informações financeiras e idade não são automaticamente classificadas como dados pessoais sensíveis pela definição do artigo 5º, inciso II, da LGPD. Sua utilização, contudo, continua exigindo cuidados de privacidade e segurança.

### 14.3. Finalidade e bases legais

Em um cenário real de fintech, o tratamento de dados para análise de risco de crédito deve possuir finalidade legítima, específica e documentada.

Entre as hipóteses legais que podem ser relevantes, conforme o contexto concreto do tratamento, estão:

| Base legal | Referência | Aplicação possível |
|---|---|---|
| Proteção do crédito | Art. 7º, X | Tratamento necessário para atividades de proteção do crédito |
| Procedimentos preliminares relacionados a contrato | Art. 7º, V | Análise realizada a pedido do titular para contratação de produto ou serviço |
| Cumprimento de obrigação legal ou regulatória | Art. 7º, II | Tratamentos necessários para atender obrigações aplicáveis à instituição |

A escolha da base legal deve ser avaliada pelo controlador de dados, considerando a finalidade, a necessidade e as características específicas de cada operação.

**Distinção importante sobre este projeto:** a implementação acadêmica utiliza um dataset público para fins de estudo e demonstração técnica. Isso não equivale a uma operação real de concessão de crédito e não representa uma definição jurídica definitiva de base legal para uma fintech.

### 14.4. Privacidade desde a concepção (Privacy by Design)

A solução considera princípios de Privacy by Design e Privacy by Default como diretrizes para uma futura aplicação produtiva.

As seguintes medidas são recomendadas:

| Medida | Objetivo | Situação no projeto |
|---|---|---|
| Minimização de dados | Utilizar apenas atributos necessários ao modelo | Aplicada na seleção das variáveis utilizadas |
| Separação dos dados | Diferenciar referência, produção simulada e rejeições | Implementada por diretórios |
| Rastreabilidade | Preservar evidências de validação e execução | Implementada por relatórios e logs |
| Controle de acesso | Restringir dados conforme a função do usuário | Não implementado |
| Criptografia | Proteger dados em trânsito e em repouso | Não implementada |
| Anonimização ou pseudonimização | Reduzir a possibilidade de identificação dos titulares | Não implementada como processo formal |
| Política de retenção | Definir prazos e critérios de descarte | Documentada como recomendação, sem automação |
| Auditoria de acesso | Registrar consultas e operações sobre dados pessoais | Não implementada |

Essas medidas devem ser aprofundadas antes de qualquer utilização com dados reais de clientes.

### 14.5. Segurança e controle de acesso

Em uma arquitetura produtiva, o acesso aos dados e aos artefatos do modelo deve seguir o princípio do menor privilégio.

Uma proposta de segregação de responsabilidades seria:

| Perfil | Responsabilidade |
|---|---|
| Engenheiro de Dados | Ingestão, preparação e qualidade dos dados |
| Cientista de Dados | Treinamento, avaliação e experimentação |
| MLOps | Implantação, monitoramento e versionamento |
| Governança e Compliance | Auditoria, políticas e conformidade |
| Administrador | Gestão de permissões e infraestrutura |

Controles recomendados:

- Autenticação individual.
- Controle de acesso baseado em função (RBAC).
- Criptografia de dados em trânsito e em repouso.
- Proteção de credenciais e segredos.
- Registro de acessos e operações relevantes.
- Separação entre ambientes de desenvolvimento, teste e produção.

Esses controles são recomendações para evolução da arquitetura e não funcionalidades integralmente implementadas neste protótipo.

### 14.6. Retenção e descarte dos dados

A retenção deve considerar a finalidade do tratamento, as obrigações legais e regulatórias aplicáveis e a necessidade de manutenção dos registros.

De acordo com os artigos 15 e 16 da LGPD, o término do tratamento pode ocorrer nas hipóteses previstas em lei, e a conservação posterior dos dados depende das condições legais aplicáveis.

Não existe neste projeto um prazo universal de retenção definido para dados financeiros. Em uma operação real, o período deverá ser formalizado pelo controlador, considerando as obrigações aplicáveis.

| Tipo de informação | Diretriz recomendada |
|---|---|
| Dados de entrada | Manter apenas durante o período necessário à finalidade e às obrigações aplicáveis |
| Dados rejeitados | Definir prazo para investigação, auditoria e descarte |
| Logs de execução | Preservar pelo período necessário à rastreabilidade e segurança |
| Métricas e relatórios | Manter conforme necessidade de auditoria e acompanhamento |
| Modelos e versões | Preservar histórico necessário à rastreabilidade das decisões |

No protótipo, os arquivos são mantidos localmente para permitir a reprodução dos experimentos. Não foi implementada uma rotina automática de expiração ou descarte.

### 14.7. Direitos dos titulares e decisões automatizadas

A LGPD estabelece direitos dos titulares, incluindo confirmação da existência de tratamento, acesso, correção e demais direitos previstos no artigo 18.

Para aplicações de Credit Scoring, também merece atenção o artigo 20, que trata do direito de solicitar a revisão de decisões tomadas unicamente com base em tratamento automatizado de dados pessoais que afetem os interesses do titular, incluindo decisões destinadas a definir seu perfil de crédito.

A instituição responsável deve estar preparada para:

- Disponibilizar informações claras sobre o tratamento.
- Explicar os critérios e procedimentos utilizados na decisão automatizada, observados os limites legais.
- Receber e tratar solicitações de revisão.
- Preservar informações necessárias à rastreabilidade da decisão.
- Investigar possíveis erros, inconsistências e resultados discriminatórios.

O protótipo não implementa um portal de atendimento ao titular nem um fluxo de revisão de decisões. Esses recursos são requisitos recomendados para uma solução produtiva.

### 14.8. Prevenção de vieses e discriminação

Modelos de Credit Scoring podem reproduzir distorções presentes nos dados históricos.

Por esse motivo, a avaliação do modelo não deve considerar somente métricas gerais de desempenho.

Recomenda-se avaliar:

- Distribuição das previsões.
- Taxas de falsos positivos e falsos negativos.
- Desempenho por grupos relevantes, quando houver base legal e metodologia apropriada.
- Estabilidade das métricas ao longo do tempo.
- Possíveis impactos discriminatórios.
- Justificativa e rastreabilidade dos critérios utilizados.

Neste projeto foram avaliadas métricas como Precision, Recall, F1-score e ROC-AUC, além das matrizes de confusão.

**Limitação:** não foi implementada uma avaliação formal de fairness por grupos demográficos ou uma auditoria de discriminação. Essa análise deve fazer parte da evolução do projeto antes de uma aplicação real em decisões de crédito.

### 14.9. Rastreabilidade e auditoria do modelo

A rastreabilidade é essencial para entender como os dados foram processados e como o modelo chegou aos resultados observados.

O projeto utiliza:

| Recurso | Evidência |
|---|---|
| MLflow | Registro de parâmetros, métricas e artefato do modelo |
| Pandera | Validação do contrato de dados |
| Arquivos JSON | Armazenamento de métricas e resultados |
| Relatórios HTML | Evidências de Data Drift e Concept Drift |
| Python Logging | Histórico das execuções operacionais |
| Estrutura de diretórios | Separação entre referência, produção e dados inválidos |

Esses recursos permitem reconstruir os experimentos realizados e identificar alterações relevantes durante a execução do pipeline.

### 14.10. Avaliação de impacto à proteção de dados

Em uma aplicação real de Credit Scoring, recomenda-se avaliar a necessidade de elaborar um Relatório de Impacto à Proteção de Dados Pessoais (RIPD), especialmente considerando a natureza das informações utilizadas e os possíveis impactos de decisões automatizadas sobre os titulares.

Essa avaliação deve considerar:

- Finalidade e necessidade do tratamento.
- Categorias de dados pessoais utilizadas.
- Riscos aos direitos e liberdades dos titulares.
- Medidas de segurança e mitigação.
- Compartilhamento de dados com terceiros.
- Critérios de retenção e descarte.
- Mecanismos de transparência e atendimento aos titulares.

A elaboração e a necessidade do RIPD devem ser avaliadas conforme o contexto e as orientações da autoridade competente. Este projeto acadêmico não constitui um RIPD formal.

### 14.11. Referências legais e institucionais

- **Lei Geral de Proteção de Dados — Lei nº 13.709/2018:**  
  https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm

- **ANPD — Direitos dos titulares de dados pessoais:**  
  https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados/direito-dos-titulares

- **Governo Digital — Solicitar revisão de decisões tomadas com base em tratamento automatizado:**  
  https://www.gov.br/governodigital/pt-br/lgpd-pagina-do-cidadao/solicitar-a-revisao-de-decisoes-tomadas-com-base-em-tratamento-automatizado-de-dados-pessoais

## 15. Limitações e considerações técnicas

Esta entrega possui caráter experimental e acadêmico. Os resultados demonstram mecanismos de validação, monitoramento e observabilidade, mas não representam uma solução de produção integrada a uma instituição financeira.

Principais limitações:

1. **Dados simulados:** as alterações de Data Drift e Concept Drift foram provocadas artificialmente para demonstrar os mecanismos de detecção.
2. **Concept Drift controlado:** os rótulos foram alterados de forma experimental, sem evidência de mudança real no comportamento de clientes.
3. **Monitoramento não contínuo:** a execução ocorre por scripts e artefatos locais, sem ingestão em tempo real.
4. **Alertas experimentais:** as regras são avaliadas sobre os resultados gerados, sem integração com ferramentas externas de notificação.
5. **Fairness:** não foi implementada uma avaliação formal de equidade entre grupos.
6. **Causalidade:** não foi implementada análise causal para explicar as causas das alterações observadas.
7. **Governança:** controles como RBAC, criptografia, anonimização formal, auditoria de acesso e descarte automatizado foram documentados como recomendações, mas não implementados.
8. **Validação temporal:** os experimentos não substituem uma validação independente com dados reais e períodos distintos.
9. **Treinamento:** o tratamento de valores ausentes antes da divisão entre treino e teste pode introduzir vazamento de informação. Uma evolução recomendada é realizar a imputação dentro do pipeline de treinamento, ajustando-a exclusivamente com os dados de treino.

Os parâmetros e limites utilizados nas simulações devem ser recalibrados com dados históricos, conhecimento do negócio e validação estatística antes de qualquer utilização produtiva.

## 16. Conclusão

O projeto demonstra como práticas de Engenharia de Dados, Machine Learning e MLOps podem ser combinadas para aumentar a confiabilidade e a rastreabilidade de modelos de Credit Scoring.

A implementação contempla validação de contratos de dados, tratamento de registros inválidos, treinamento de um modelo baseline, registro de experimentos, detecção estatística de Data Drift, simulação de Concept Drift e consolidação de indicadores operacionais.

Os experimentos evidenciaram que alterações nas distribuições dos dados e na relação entre atributos e variável alvo podem comprometer o desempenho do modelo, especialmente na identificação de clientes inadimplentes.

A inclusão de diretrizes de governança e LGPD reforça a importância de considerar privacidade, transparência, segurança e prevenção de discriminação durante todo o ciclo de vida de soluções de inteligência artificial aplicadas ao crédito.

Dessa forma, a observabilidade deixa de ser apenas uma atividade técnica de monitoramento e passa a ser um componente essencial para a gestão de riscos, a governança e a evolução responsável de modelos de Machine Learning.