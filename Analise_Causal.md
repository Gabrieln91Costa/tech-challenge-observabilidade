Análise causal da degradação do modelo de Credit Scoring
1. Objetivo
Investigar a queda de desempenho observada no experimento de Concept
Drift, identificando:
- quais variáveis e condições foram associadas à degradação;
- por que o Recall caiu de 80,48% para 39,21%;
- como o aumento da inadimplência simulada afetou as previsões;
- quais ações de monitoramento e mitigação podem reduzir o problema.
Nota metodológica: este projeto utiliza uma simulação controlada.
Portanto, os resultados permitem explicar o mecanismo da degradação
dentro do experimento, mas não comprovam relações causais em uma
operação financeira real. A análise distingue os fatos medidos das
interpretações e hipóteses.

2. Contexto do experimento
O modelo de referência é um Random Forest treinado com dados históricos
do conjunto Give Me Some Credit. Para avaliar a degradação, o modelo foi
mantido congelado: não houve retreinamento entre as duas avaliações.

A simulação de Concept Drift foi construída da seguinte forma:
1. Foram selecionados 30.000 registros da base de referência.
2. Foi identificado um segmento de maior risco, definido por:
   - DebtRatio igual ou superior ao percentil 75 da amostra
     (aproximadamente 0,8652);
   - MonthlyIncome igual ou inferior à mediana (aproximadamente
     5.400).
3. Esse segmento continha 7.255 registros.
4. Em 50% dos registros elegíveis cujo rótulo original indicava
   ausência de inadimplência (SeriousDlqin2yrs = 0), o rótulo foi
   alterado para inadimplência (SeriousDlqin2yrs = 1).
5. Foram alterados 3.402 rótulos. Os valores das variáveis
   preditoras foram preservados.
Assim, o experimento alterou a relação entre as características dos
clientes (X) e o resultado observado (Y), sem modificar a
distribuição das variáveis de entrada. Esse cenário representa uma
simulação de mudança no conceito aprendido pelo modelo: Concept
Drift.
3. Evidências observadas
3.1 Comparação das métricas
  Métrica       Antes da simulação   Após a simulação          Variação
  Accuracy                  83,06%             75,21%        -7,85 p.p.
Precision                 25,15%             33,50%        +8,35 p.p.
Recall                    80,48%             39,21%   -41,27 p.p.
F1-score                  38,33%             36,13%        -2,20 p.p.
ROC-AUC                   90,18%             59,95%   -30,23 p.p.

O resultado mais preocupante é a redução do Recall. O modelo passou a
identificar uma parcela muito menor dos clientes que efetivamente
pertencem à classe de inadimplência na base simulada.

A queda da ROC-AUC também é relevante: ela indica perda expressiva da
capacidade de ordenação/discriminação entre as classes no cenário
simulado.
3.2 Matriz de confusão
  Resultado                       Antes        Após
  Verdadeiros negativos (TN)     23.339      20.461
Falsos positivos (FP)           4.699       4.175
Falsos negativos (FN)             383   3.261
Verdadeiros positivos (TP)      1.579       2.103

O número de falsos negativos aumentou de 383 para 3.261, ou seja,
foram 2.878 falsos negativos adicionais no cenário simulado.

Em um processo de concessão de crédito, um falso negativo neste contexto
representa um cliente que foi classificado pelo modelo como não
inadimplente, embora o rótulo observado indique inadimplência. Esse erro
pode aumentar a exposição ao risco de crédito.
3.3 Alteração da taxa de inadimplência
  Indicador                            Antes     Após
  Taxa de inadimplência na amostra     6,54%   17,88%

A taxa de inadimplência simulada aumentou 11,34 pontos percentuais.
O modelo, entretanto, permaneceu com os mesmos parâmetros e padrões
aprendidos durante o treinamento.
4. Quais variáveis contribuíram para a degradação?
É importante separar dois experimentos realizados no projeto.
4.1 Evidência de Data Drift
Na simulação de produção (credit_production.csv), foram aplicadas as
seguintes alterações:
- MonthlyIncome: redução de 25%;
- DebtRatio: aumento de 35%;
- RevolvingUtilizationOfUnsecuredLines: aumento de 25%.
A análise com Evidently identificou drift em 2 das 10 variáveis
avaliadas:
  Variável                                          Distância reportada Resultado
(Wasserstein normalizada) 
  MonthlyIncome                                              0,083949 Drift detectado

  DebtRatio                                                  0,076569 Drift detectado

  RevolvingUtilizationOfUnsecuredLines                       0,014316 Sem drift detectado
Demais variáveis                         Abaixo do critério utilizado Sem drift detectado
O resultado indica mudança estatística nas distribuições de renda mensal
e relação dívida/renda no conjunto de produção simulado.
4.2 Evidência de Concept Drift
No experimento de Concept Drift, as variáveis preditoras foram
preservadas. A mudança ocorreu nos rótulos de inadimplência,
especialmente no segmento definido por DebtRatio elevado e
MonthlyIncome reduzida.

Portanto:
- DebtRatio e MonthlyIncome foram utilizadas para definir o
  segmento no qual a alteração de rótulos foi aplicada;
- a mudança de rótulos nesse segmento elevou a taxa de inadimplência
  da amostra;
- o modelo congelado não foi atualizado para refletir a nova relação
  entre as características dos clientes e a inadimplência.
Limite da conclusão: a simulação demonstra que a alteração dos
rótulos nesse segmento foi acompanhada pela queda das métricas. Ela não
demonstra que DebtRatio ou MonthlyIncome, isoladamente, causaram a
queda. Também não foi executado um método de atribuição causal ou uma
análise de importância de variáveis para quantificar a contribuição
individual de cada atributo.
5. Por que o Recall caiu de 80,48% para 39,21%?
O Recall é calculado por:

\(Recall = `\frac{TP}{TP + FN}`{=tex}\)

Ele mede a proporção de inadimplentes reais que o modelo conseguiu
identificar.

No cenário anterior à simulação, o modelo identificava 1.579 dos 1.962
casos positivos, resultando em Recall de aproximadamente 80,48%.

Após a alteração dos rótulos, o modelo identificou 2.103 dos 5.364 casos
positivos, resultando em Recall de aproximadamente 39,21%.

O mecanismo observado foi:
1. O modelo foi treinado com padrões históricos nos quais determinadas
   combinações de características estavam associadas a uma
   probabilidade de inadimplência.
2. Na simulação, parte dos clientes do segmento de maior relação
   dívida/renda e menor renda passou a receber o rótulo de
   inadimplência.
3. Como o modelo permaneceu congelado, suas previsões continuaram
   baseadas nos padrões anteriores.
4. Parte dos novos casos positivos não foi identificada como
   inadimplente.
5. Os falsos negativos aumentaram de 383 para 3.261, reduzindo o
   Recall.
A queda, portanto, é compatível com uma mudança na relação entre as
variáveis de entrada e o resultado observado, sem adaptação do modelo a
essa nova realidade.
6. Como a alteração da inadimplência afetou as previsões?
O aumento da taxa de inadimplência de 6,54% para 17,88% elevou a
quantidade de casos positivos que o modelo precisava identificar.

Entretanto, a alteração foi feita nos rótulos, não nas variáveis
utilizadas pelo modelo para gerar as previsões. Como consequência, as
previsões do modelo congelado não passaram a refletir automaticamente o
aumento da inadimplência.

Isso explica por que a quantidade de falsos negativos cresceu
significativamente. O modelo classificou como não inadimplentes diversos
registros que, na base simulada, passaram a ser considerados
inadimplentes.

A Precision aumentou de 25,15% para 33,50%, mas esse aumento não
significa que o modelo melhorou de forma geral. O Recall caiu fortemente
e a ROC-AUC também apresentou degradação. Em um problema de crédito,
avaliar somente a Precision poderia esconder o aumento de clientes
inadimplentes não identificados.
7. Ações recomendadas para reduzir o problema
7.1 Monitoramento contínuo
- Monitorar a taxa de inadimplência observada e sua evolução por
  período e segmento.
- Acompanhar Recall, Precision, F1-score, ROC-AUC e matriz de confusão
  após a chegada dos rótulos reais.
- Monitorar separadamente Data Drift nas variáveis de entrada e
  Concept Drift no desempenho do modelo.
- Criar alertas para quedas relevantes de Recall e ROC-AUC, com
  limites definidos a partir do histórico e do apetite de risco da
  instituição.
7.2 Investigação por segmento
- Avaliar métricas por faixas de DebtRatio e MonthlyIncome.
- Verificar se a queda de Recall está concentrada em grupos
  específicos.
- Comparar a taxa de falsos negativos entre períodos e segmentos.
- Investigar possíveis mudanças no perfil dos clientes, nas políticas
  de concessão e nas condições econômicas.
7.3 Revisão do modelo e do processo decisório
- Reavaliar a representatividade e a atualidade da base de
  treinamento.
- Considerar retreinamento somente após validação da qualidade dos
  novos rótulos e dos dados.
- Comparar o modelo atualizado com o modelo atual em uma amostra
  temporal independente.
- Avaliar a calibração das probabilidades e os limiares de decisão.
- Definir o limiar de classificação considerando o custo de falsos
  negativos e falsos positivos, além das regras de negócio e de
  governança.
Ajustar o limiar pode alterar o equilíbrio entre Recall e Precision, mas
não corrige, por si só, uma mudança estrutural na relação entre as
variáveis e a inadimplência.
7.4 Governança e segurança da decisão
- Registrar versões do modelo, dados, métricas, limiares e decisões de
  retreinamento.
- Manter trilha de auditoria para as alterações e aprovações.
- Avaliar impactos sobre diferentes segmentos de clientes e possíveis
  vieses.
- Manter mecanismos de revisão e contestação compatíveis com a
  legislação e as políticas da instituição.
- Não automatizar retreinamento em produção sem validação, testes e
  aprovação.
8. Conclusão
O experimento demonstrou que a alteração controlada dos rótulos de
inadimplência em um segmento de maior relação dívida/renda e menor renda
foi acompanhada por uma degradação relevante do modelo congelado.

A taxa de inadimplência simulada aumentou de 6,54% para 17,88%. Nesse
cenário, o Recall caiu de 80,48% para 39,21%, enquanto os falsos
negativos aumentaram de 383 para 3.261 e a ROC-AUC caiu de 90,18% para
59,95%.

A principal explicação experimental é a mudança da relação entre as
características dos clientes e o resultado de inadimplência, sem
atualização do modelo. As variáveis DebtRatio e MonthlyIncome
definiram o segmento afetado pela simulação; no experimento separado de
Data Drift, ambas também apresentaram mudança estatística de
distribuição.

Os resultados reforçam a necessidade de monitorar dados e desempenho
continuamente, investigar degradações por segmento e estabelecer
critérios controlados para revisão, recalibração ou retreinamento do
modelo.

Ressalva: trata-se de uma simulação acadêmica controlada. Para
afirmar causalidade em uma operação real, seria necessário utilizar
dados reais, desenho de avaliação apropriado e investigação adicional de
fatores externos. Os resultados deste projeto não devem ser
interpretados como prova de que as variáveis analisadas causam
inadimplência ou determinam, isoladamente, o desempenho do modelo.
9. Artefatos do projeto utilizados
- data/production/credit_concept_drift.csv --- amostra com rótulos
  alterados na simulação.
- reports/observability/concept_drift_simulation.json --- resumo da
  simulação e quantidade de rótulos alterados.
- reports/observability/concept_drift_evaluation.json --- métricas
  antes e depois da simulação.
- reports/observability/concept_drift_report.html --- relatório HTML
  da avaliação de Concept Drift.
- reports/data_drift/data_drift_metrics.json --- métricas
  estatísticas do experimento de Data Drift.
- reports/data_drift/data_drift_report.html --- relatório HTML de
  Data Drift.
- models/credit_model.pkl --- modelo congelado utilizado na
  avaliação.