# Revisão de literatura e estado da arte — previsão de demanda de antimicrobianos (Entrega 5)

> Busca intensiva na literatura recente e no estado da arte para fundamentar cada
> decisão metodológica do projeto (formulação, unidade, métrica, baseline, modelo,
> avaliação). Cada bloco mapeia o achado da literatura à decisão correspondente da
> equipe. As fontes estão linkadas; o conteúdo foi parafraseado/resumido para
> conformidade de licenciamento, preservando os fatos e números.
>
> *Content was rephrased for compliance with licensing restrictions.*

---

## 0. Pergunta de pesquisa

Como a literatura recente resolve a previsão de demanda de medicamentos (em especial
antimicrobianos) em farmácia hospitalar, quando há (a) muitos itens em escalas
diferentes, (b) horizonte curto de decisão, (c) poucos meses de histórico e (d)
padrões de demanda estáveis ou intermitentes? E o que isso diz sobre a escolha de
baseline, métrica e modelo?

---

## 1. O problema exato existe na literatura e tem a mesma forma

- **Hospital trust network features inform ML for medicines demand forecasting**
  (Springer, *Discover Applied Sciences*, 2026) — afirma diretamente que farmácias
  hospitalares precisam de previsões de **horizonte curto** que continuem úteis
  através de **milhares de medicamentos com volumes de saída muito diferentes**.
  É exatamente a nossa situação (antimicrobiano × mês, escalas heterogêneas).
  [link](https://link.springer.com/article/10.1007/s44163-026-01956-z)
- **Optimising hospital pharmacy inventory for high-value, intermittent-demand drugs
  (LightGBM-Tweedie)** (*British Journal of Healthcare Management / lifescience*, 2021)
  — foca em medicamentos de demanda "lumpy/erratic", onde o pedido manual costuma
  falhar por capacidade limitada de previsão. Avalia com **wMAPE** (= WAPE) e
  **estratifica por padrão de demanda**, comparando contra **naïve, ARIMA, Random
  Forest e LSTM**.
  [link](https://www.lifescience.net/publications/2265442/optimising-hospital-pharmacy-inventory-for-high-va/)

**Mapa p/ nossa decisão:** valida (i) a unidade por item com escalas diferentes,
(ii) a métrica WAPE/wMAPE como padrão da área, (iii) a prática de comparar o modelo
contra um baseline naïve e estratificar por padrão de demanda. Exatamente o que
fizemos.

---

## 2. Antimicrobianos especificamente: ARIMA/ETS sobre consumo em DDD é o padrão

- **Long-term forecast for antibacterial drug consumption in Germany using ARIMA**
  (*Infection*, PubMed 2025) — previsão de consumo de antibacterianos por **DDD**
  (dose diária definida) com modelos **ARIMA** por princípio ativo.
  [link](https://pubmed.ncbi.nlm.nih.gov/39754681/)
- **Observed and Forecasted Antibiotic Consumption by ATC/DDD Metrics** (*Antibiotics*,
  MDPI 2026) — modela uso de antibióticos em **DDD por 100 leitos-dia**, por classe
  (penicilinas, cefalosporinas, fluoroquinolonas…); reporta erro percentual absoluto
  de 2,16% (2024) e 4,61% (2025), **MAPE bianual de 3,39%** para cefalosporinas.
  [link](https://www.mdpi.com/2079-6382/15/8/805)
- **ARIMA no monitoramento do uso de antibióticos em ambulatório** (PubMed 2020) —
  erro absoluto médio de 0,76% na taxa de uso.
  [link](https://pubmed.ncbi.nlm.nih.gov/32145754/)

**Mapa p/ nossa decisão:** a literatura de antimicrobianos trabalha **por item/classe**,
em **séries mensais**, com métricas de erro percentual — e obtém erros baixos quando a
série é **longa e estável** (anos de histórico). Isso contextualiza por que, com 19
meses, não alcançamos os MAPEs de 2–5% desses estudos: eles têm muito mais histórico.
Reforça nossa limitação declarada (janela curta) como fator causal, não falha de método.

---

## 3. Caso brasileiro / SUS: ETS vence e o ganho vem da comparação com o baseline

- **Comparing Time Series Techniques for Medication Demand Forecasting** (De Meo &
  Gonçalves, *DGO 2025*, Secretaria de Saúde de São Paulo + USP) — prevê demanda de
  **captopril** nas UBS de São Paulo com **ETS e ARIMA**, janeiro/2018 a março/2023,
  comparando por **MAPE**. O **ETS atingiu MAPE de 2,26%**, contra **77,97% da
  metodologia vigente** da prefeitura.
  [link](https://proceedings.open.tudelft.nl/DGO2025/article/download/1013/1047/1419)

**Mapa p/ nossa decisão:** é o espelho brasileiro do nosso projeto (demanda de
medicamento no sistema público de saúde). Dois aprendizados diretos: (i) **o valor
não está no modelo sofisticado, mas na comparação honesta contra o baseline** — o
ganho relatado é "modelo vs prática atual"; (ii) eles tinham **5+ anos de dados** para
um único item de altíssimo volume e demanda regular — condições que tornam ETS muito
preciso. Nosso recorte (muitos itens, 19 meses) é estruturalmente mais difícil.

---

## 4. Métrica: WAPE/wMAPE é o padrão; MAPE falha com itens pequenos; MASE mede *skill*

- **Hyndman & Koehler (2006), "Another look at measures of forecast accuracy"** — texto
  fundacional. Mostra as falhas do MAPE (explode/indefinido quando o real é pequeno ou
  zero) e propõe o **MASE (Mean Absolute Scaled Error)** como padrão para comparar
  acurácia entre séries: é o MAE do modelo **dividido pelo MAE de um baseline naïve**.
  MASE < 1 ⇒ o modelo supera o baseline; MASE ≥ 1 ⇒ não supera.
  [link](https://www.researchgate.net/publication/5055536_Another_Look_at_Forecast_Accuracy_Metrics_for_Intermittent_Demand)
  · [nota IJF](https://pure.eur.nl/ws/portalfiles/portal/47633436/A-note-on-the-MASE-Revision-for-IJF.pdf)
- **WAPE vs MAPE na prática de demand planning** — WAPE pondera o erro pelo volume do
  item, alinhando a acurácia ao impacto real em estoque; MAPE trata todos os itens
  igualmente, então um erro de 50% num item de baixíssimo volume pesa igual ao de um
  item carro-chefe.
  [areteinc](https://www.areteinc.com/blog/wape-vs-mape-for-forecast-accuracy-reporting/)
  · [predicthq](https://docs.predicthq.com/getting-started/guides/forecasts-api-guides/understanding-forecast-accuracy-metrics)

**Mapa p/ nossa decisão:** justifica tecnicamente **WAPE como métrica principal** (itens
de escalas muito diferentes; decisão dominada por itens de alto volume) e **MAPE médio
como secundária** (interpretável, mas infla com itens pequenos). E dá nome ao que
fizemos: comparar modelo vs baseline é, em espírito, o **MASE** — nosso modelo tem
**MASE ≈ 1** (empata com o naïve), que é a forma rigorosa de dizer "não supera o baseline".

---

## 5. Demanda intermitente: Croston/SBA/TSB e "zero às vezes é o melhor palpite"

- **Croston (1972), SBA (Syntetos-Boylan), TSB** — métodos padrão para demanda
  intermitente: suavizam separadamente o **tamanho** da demanda e o **intervalo entre
  demandas**. A SBA é o "de facto standard", corrigindo o viés de alta do Croston.
  [metricgate/SBA](https://metricgate.com/docs/intermittent-demand-syntetos/)
  · [Kourentzes 2014](https://eprints.lancs.ac.uk/69640/1/Kourentzes_2013_Intermittent_Optimisation.pdf)
- **Kourentzes (2014)** — alerta que **métricas convencionais de acurácia são
  inadequadas** para seleção de modelo em demanda intermitente, e que mesmo a seleção
  automática **raramente supera substancialmente modelos únicos**.
- **Tese LIACS (2026), local vs global em séries intermitentes** — achado contundente:
  em grão semanal, **"nenhum modelo supera prever zero"**; e com pesos iguais entre
  SKUs, a vantagem do LightGBM encolhe.
  [link](https://theses.liacs.nl/pdf/2025-2026-MciszKPKamil.pdf)

**Mapa p/ nossa decisão:** embasa termos **descartado o grão item×setor×mês** (53,6%
zeros — intermitente/inviável) e ficado nos **20 itens densos**. E ensina a lição dura
que vimos empiricamente: em regime esparso/estável, **o baseline trivial é um competidor
forte e legítimo** — a literatura trata "não superar o baseline" como resultado comum,
não anomalia.

---

## 6. Competições M (M4/M5): os achados que sustentam nossa escolha de modelo

- **M4 (Makridakis et al., 2018)** — dos 17 métodos mais acurados, **12 eram combinações
  de abordagens majoritariamente estatísticas**; métodos puros de ML ficaram atrás.
  [PDF](https://www.researchgate.net/profile/Spyros-Makridakis/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward)
- **M5 (2022)** — em dados de varejo hierárquicos ricos (Walmart, 42.840 séries),
  métodos de ML/gradient boosting **superaram** os benchmarks estatísticos; mas os
  autores destacam que **métodos estatísticos locais simples seguem competitivos** em
  alta granularidade e nas caudas da incerteza.
  [predição M5](https://www.researchgate.net/publication/356338301_Predictinghypothesizing_the_findings_of_the_M5_competition)
  · [conclusões M5](https://purehost.bath.ac.uk/ws/portalfiles/portal/233855891/IJF_2022_M5_Conclusions_post_print_.pdf)
- **Síntese das competições M** — três veredictos consolidados: **(1) métodos simples
  são difíceis de bater; (2) combinar previsões ajuda de forma confiável; (3) acurácia
  tem de ser julgada honestamente em dados que o modelo nunca viu.**
  [resumo](https://r-statistics.co/M-Competition-Lessons-in-R.html)

**Mapa p/ nossa decisão:** o ML só vence quando há **muitos dados e estrutura
cross-series** (M5/Walmart); com **poucos dados e sem estrutura temporal** (nosso caso),
o esperado pela literatura é que **modelos simples sejam difíceis de bater** — que é
exatamente o que medimos. Justifica escolher **regressão linear sobre o nível do item**
em vez de árvores/LSTM, e validar no **hold-out temporal** (achado 3 das competições M).

---

## 7. Mais dados ⇒ ML melhora; com poucos dados, o simples ganha

- **Learning-curve / sample size e performance de ML em forecasting** (arXiv 1909.13316)
  — a performance **relativa** dos métodos de ML melhora **conforme o tamanho da amostra
  cresce**. Em amostras pequenas, não compensam.
  [link](https://arxiv.org/pdf/1909.13316v1.pdf)
- **"Do We Really Need Deep Learning Models for Time Series Forecasting?"** (arXiv
  2101.02118) — gradient boosting com engenharia de atributos simples rivaliza/supera
  deep learning em muitos benchmarks; o custo-benefício do modelo complexo é questionável.
  [link](https://arxiv.org/html/2101.02118v2)

**Mapa p/ nossa decisão:** com **15 meses de treino**, estamos no regime de "amostra
pequena" onde a literatura prevê que **ML sofisticado não compensa**. Nossa escolha de
um modelo linear interpretável é a recomendada para o regime de dados que temos.

---

## 8. "Forecastability": existe medida formal para dizer que uma série é pouco previsível

- **How to Assess Forecastability (entropia amostral)** — séries com **baixa entropia
  (alta regularidade)** correlacionam com **baixo erro fora da amostra (MASE)**; a
  recíproca indica séries pouco previsíveis.
  [link](https://www.researchgate.net/publication/46533832_How_to_Assess_Forecastability)
- **Coeficiente de variação como (imperfeita) medida de forecastability** — CV = desvio
  padrão / média; útil, mas **não se aplica a dados intermitentes** e **ignora a
  sequência** (uma série em tendência perfeita pode ter CV alto e ainda ser previsível).
  [arkieva](https://arkieva.com/blog/do-you-use-coefficient-of-variation-to-determine-forecastability/)
- **Auto-mutual information (AMI) e forecastability por horizonte** (arXiv 2601.10006)
  — mede a **redução de incerteza que o passado fornece**; para séries mensais o sinal
  é fraco/dependente do método-probe (correlação de AMI com erro de −0,12 a −0,26).
  [link](https://arxiv.org/pdf/2601.10006v1.pdf)

**Mapa p/ nossa decisão:** dá **suporte teórico formal** ao nosso diagnóstico empírico.
Medimos autocorrelação lag-1 ≈ 0 (mensal) e −0,11 (semanal): em linguagem de
forecastability, a série tem **baixa informação do passado sobre o futuro além do nível
médio** — por isso o melhor preditor honesto colapsa na média do item. Não é falha de
modelagem; é uma **propriedade mensurável da série**.

---

## 9. Baseline forte e avaliação honesta são o estado da arte, não o contrário

- **ConformalNaive: um baseline training-free é surpreendentemente forte** (arXiv
  2606.09473) — em 2.217 séries reais de 9 fontes públicas, um intervalo naïve sem
  treino **bate métodos aprendidos** e só perde para métodos adaptativos que rastreiam
  mudança de distribuição.
  [link](https://arxiv.org/html/2606.09473v1)
- **Foundation models "papagaiam" o contexto** (arXiv 2505.11349) — modelos de fundação
  de série temporal frequentemente preveem por **cópia do contexto** ou **convergência à
  média**; um "context parroting" ingênuo supera modelos de fundação líderes a fração do
  custo.
  [link](https://arxiv.org/html/2505.11349v3)

**Mapa p/ nossa decisão:** o estado da arte **leva o baseline a sério** e trata "o
baseline é difícil de bater" como achado publicável. Nossa postura — comparar sempre na
mesma partição, reportar empate com transparência, distinguir sinal de sobreajuste via
backtest de origem expansível — está alinhada à prática mais atual, não aquém dela.

---

## 10. Síntese: o que a literatura nos autoriza a afirmar na Entrega 5

1. **Formulação/unidade:** prever **por item (antimicrobiano × mês)** com escalas
   diferentes é o enquadramento canônico da farmácia hospitalar (§1, §2).
2. **Métrica:** **WAPE principal** (ponderada por volume, alinhada ao estoque) e MAPE
   secundária; comparar modelo vs baseline é, em espírito, **MASE** (§4).
3. **Escopo denso:** excluir o grão esparsíssimo (item×setor, 53% zeros) é coerente com
   a literatura de demanda intermitente, onde "prever zero" pode ser imbatível (§5).
4. **Modelo:** com **15 meses de treino e sem estrutura temporal**, a literatura prevê
   que **modelos simples são difíceis de bater** e que ML só compensa com muitos dados e
   estrutura cross-series — logo, **regressão linear sobre o nível do item** é a escolha
   fundamentada (§6, §7).
5. **Resultado honesto (empate):** não superar um baseline forte é **resultado comum e
   publicável** no estado da arte; a série tem **baixa forecastability mensurável** (§8,
   §9). Relatar o empate com rigor é a prática recomendada, não um demérito.
6. **Comparação brasileira (captopril/ETS):** o valor de projetos assim está na
   **comparação honesta com a prática vigente** e depende fortemente de **histórico
   longo** — o que explica nossos limites e aponta o próximo passo (mais meses de dados)
   (§3).

---

## Referências (acesso em 2026-10-29)

1. Hospital trust network features inform ML for medicines demand forecasting — Springer, 2026. https://link.springer.com/article/10.1007/s44163-026-01956-z
2. Optimising hospital pharmacy inventory (LightGBM-Tweedie) — 2021. https://www.lifescience.net/publications/2265442/optimising-hospital-pharmacy-inventory-for-high-va/
3. Long-term forecast for antibacterial drug consumption in Germany (ARIMA) — PubMed, 2025. https://pubmed.ncbi.nlm.nih.gov/39754681/
4. Observed and Forecasted Antibiotic Consumption by ATC/DDD — MDPI Antibiotics, 2026. https://www.mdpi.com/2079-6382/15/8/805
5. ARIMA no uso de antibióticos em ambulatório — PubMed, 2020. https://pubmed.ncbi.nlm.nih.gov/32145754/
6. De Meo & Gonçalves, Comparing Time Series Techniques for Medication Demand Forecasting (captopril, SP) — DGO 2025. https://proceedings.open.tudelft.nl/DGO2025/article/download/1013/1047/1419
7. Hyndman & Koehler, Another look at measures of forecast accuracy (MASE) — IJF 2006. https://www.researchgate.net/publication/5055536_Another_Look_at_Forecast_Accuracy_Metrics_for_Intermittent_Demand
8. WAPE vs MAPE for forecast accuracy — Arete/PredictHQ. https://www.areteinc.com/blog/wape-vs-mape-for-forecast-accuracy-reporting/
9. Kourentzes, Optimisation of intermittent demand methods — 2014. https://eprints.lancs.ac.uk/69640/1/Kourentzes_2013_Intermittent_Optimisation.pdf
10. Syntetos-Boylan Approximation (SBA) — metricgate. https://metricgate.com/docs/intermittent-demand-syntetos/
11. Local vs global models on intermittent series (LIACS thesis) — 2026. https://theses.liacs.nl/pdf/2025-2026-MciszKPKamil.pdf
12. Makridakis et al., The M4 Competition: results, findings, conclusions — IJF 2018. https://www.researchgate.net/profile/Spyros-Makridakis/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward
13. The M5 Competition: conclusions — IJF 2022. https://purehost.bath.ac.uk/ws/portalfiles/portal/233855891/IJF_2022_M5_Conclusions_post_print_.pdf
14. Lessons from the M Competitions — r-statistics. https://r-statistics.co/M-Competition-Lessons-in-R.html
15. Sample size and relative ML forecasting performance — arXiv 1909.13316. https://arxiv.org/pdf/1909.13316v1.pdf
16. Do We Really Need Deep Learning Models for Time Series Forecasting? — arXiv 2101.02118. https://arxiv.org/html/2101.02118v2
17. How to Assess Forecastability (sample entropy / MASE) — 2009. https://www.researchgate.net/publication/46533832_How_to_Assess_Forecastability
18. Coefficient of variation and forecastability — Arkieva. https://arkieva.com/blog/do-you-use-coefficient-of-variation-to-determine-forecastability/
19. Auto-mutual information and horizon-specific forecastability — arXiv 2601.10006. https://arxiv.org/pdf/2601.10006v1.pdf
20. A training-free conformal interval is a strong baseline — arXiv 2606.09473. https://arxiv.org/html/2606.09473v1
21. Foundation models often forecast by context parroting — arXiv 2505.11349. https://arxiv.org/html/2505.11349v3

*Revisão redigida pela equipe a partir de busca na literatura; trechos de fontes foram
parafraseados para conformidade de licenciamento. Nenhuma afirmação numérica foi
inventada — todas vêm das fontes citadas.*
