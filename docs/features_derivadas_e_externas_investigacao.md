# Investigação de features derivadas internas e dados externos — regressão de demanda (Entrega 5)

> Análise **somente leitura**. Nenhum módulo de produção foi alterado
> (`ingestion/`, `processing/`, `features/`, `modeling/baseline.py`,
> `modeling/baseline_by_item.py` intactos). Todo o código desta investigação vive em
> `src/huforecast/analysis/external_sources.py` (download real dos dados públicos) e
> `src/huforecast/analysis/derived_external_investigation.py` (features + avaliação),
> e roda com `python -m huforecast.analysis.derived_external_investigation`
> (semente fixa `SEED = 42`, sem caminhos absolutos, via `huforecast.config`).
> Todos os números vêm de execução real sobre `data/raw/Planilha CCIH 2026 Atualizado.xlsx`
> (7.261 tratamentos) e sobre dados públicos **efetivamente baixados** da API InfoDengue
> (João Pessoa/PB, geocode 2507507), cacheados em `data/external/`.
> Partição e métrica idênticas ao resto do semestre: treino 2025-03→2026-05 (15 meses),
> teste 2026-06→2026-09 (4 meses), mês 2026-10 excluído; WAPE principal, MAPE médio por
> item secundário.

---

## 1. Resumo executivo (a resposta primeiro)

**Houve sinal honesto que supere o baseline de forma material e consistente? NÃO.**

Esta terceira investigação tentou extrair mais sinal para a regressão por duas frentes
— features derivadas internas (defasadas em t-1) e dados públicos externos (clima e
epidemiologia de João Pessoa). O resultado honesto é:

- **Frente 1 (derivadas internas):** nenhuma feature derivada honesta supera o baseline
  de forma consistente. No nível **item × mês (20 densos, baseline WAPE 22,91%)**, o
  melhor número honesto aparece com `media+pressão+posologia` sob GradientBoosting
  (**WAPE 21,61%**, ~1,3 pp abaixo do baseline), mas o **mesmo conjunto sob
  LinearRegression piora para 23,21%** — a melhora não é estável entre estimadores, o
  que é a assinatura de ruído, não de sinal. As features de média móvel (`rollmean2/3`)
  e o nível histórico do item dominam a importância, confirmando o achado das
  investigações anteriores: o melhor preditor honesto é o **nível médio do item**.

- **Frente 2 (dados externos):** no nível **item × mês**, as variáveis externas têm
  correlação **praticamente nula** com a demanda (|Spearman| ≤ 0,03) e **não ajudam**.
  No nível **total mensal agregado (baseline MAPE 9,55%)**, na janela fixa de 4 meses
  `media+clima_lag1` (regressão linear) parece vencer com **MAPE 7,89% / WAPE 7,31%**
  (~1,7 pp de MAPE abaixo do baseline). **Esse ganho, porém, NÃO é robusto:** num
  backtest de origem expansível (10 previsões um-passo-à-frente, o uso realista mês a
  mês), o mesmo modelo **piora para MAPE 14,74%** contra **10,21%** do baseline da média.
  A aparente vitória na janela fixa é **sobreajuste a 4 meses específicos**, exatamente o
  tipo de diferença que o próprio brief classifica como ruído.

**Veredito para a Entrega 5 (número exato):** o melhor modelo **honesto** de regressão
continua sendo o baseline — a **média histórica do item** (regressão linear sobre
`media_item_ate_tm1`), que **empata** com o baseline trivial por item em **WAPE ≈ 22,5%
vs 22,91%** no nível item × mês, e no nível agregado a **média (MAPE 9,55%)** permanece
imbatível de forma robusta. Nem as derivadas internas nem os dados externos trouxeram
ganho honesto sustentável. **Mantém-se a recomendação das investigações anteriores.**

Os dados externos ainda assim renderam **insights descritivos úteis ao demandante**
(seção 6), mesmo não virando preditor.

---

## 2. O que foi baixado de verdade (Frente 2 — fontes)

Módulo: `src/huforecast/analysis/external_sources.py`. Cache: `data/external/`
(`joaopessoa_clima_epidemia_mensal.csv` + `FONTES.json` com URLs, status e erros).

| Fonte | URL | Status | O que entregou |
|---|---|---|---|
| **InfoDengue (dengue)** | `info.dengue.mat.br/api/alertcity?geocode=2507507&disease=dengue…` | **OK** | 95 semanas → 23 meses: casos de dengue **+ clima** (temp/umidade) de João Pessoa |
| **InfoDengue (chikungunya)** | `…&disease=chikungunya…` | **OK** | 95 semanas → 23 meses de casos de chikungunya |
| **Clima (InfoDengue)** | mesma resposta da dengue | **OK** | tempmed/min/max e umidmed/min/max mensais de João Pessoa |
| **INMET (estação A320)** | `apitempo.inmet.gov.br/estacao/2025-03-01/2025-03-31/A320` | **FALHOU** | corpo vazio — a `apitempo` não serve dados recentes dessa estação sem token/export do portal BDMEP |
| **InfoGripe / Fiocruz SRAG** | `gitlab.procc.fiocruz.br/mave/repo/-/raw/master/Dados/InfoGripe/serie_temporal_com_estimativas_recentes.csv.gz` | **FALHOU** | timeout repetido (>120 s) no endpoint raw do GitLab |

**Decisão honesta:** como o INMET e o InfoGripe não baixaram, o **clima veio da própria
resposta da InfoDengue** (derivado de satélite/estação para o geocode de João Pessoa), que
é real e específico do município. As falhas estão documentadas em código
(`fetch_inmet_attempt`, `fetch_infogripe_attempt`) e em `data/external/FONTES.json` — nada
foi inventado. A hipótese de SRAG respiratória (mais diretamente ligada a antibióticos que
arboviroses) **não pôde ser testada** por indisponibilidade da fonte; fica registrada como
limitação e próximo passo.

Cobertura da série externa (mensal): **2024-12 a 2026-10**, cobrindo integralmente a janela
de modelagem 2025-03 → 2026-09.

---

## 3. Classificação de vazamento por feature (requisito obrigatório)

Regra: para prever o mês **t** só vale informação até o fim de **t-1**. Dicionário
completo em `build_leakage_table` (`derived_external_investigation.py`).

### Features derivadas internas

| Feature | Classe | Justificativa |
|---|---|---|
| `media_item_ate_tm1` | **PERMITIDA** | Média histórica do item até t-1 (nível). |
| `qtd_total_lag1` | **PERMITIDA** | Demanda do item no mês anterior (persistência). |
| `rollmean2_lag1`, `rollmean3_lag1` | **PERMITIDA** | Médias móveis de 2/3 meses da demanda até t-1. |
| `tendencia_local` | **PERMITIDA** | `demanda(t-1) − demanda(t-2)`; fechada no corte. |
| `mes_do_ano`, `trimestre` | **PERMITIDA** | Calendário; conhecido para qualquer mês futuro. |
| `doses_dia_item_lag1` | **PERMITIDA** | Perfil posológico médio do item em t-1 (`12/12`→2 doses/dia). |
| `hosp_ativos_lag1` | **PERMITIDA** | Pressão hospitalar (nº de tratamentos ativos) em t-1. |
| `hosp_diversidade_lag1` | **PERMITIDA** | Diversidade de antimicrobianos em uso em t-1. |

### Features externas

| Feature | Classe | Justificativa |
|---|---|---|
| `temp*_lag1`, `umid*_lag1` | **PERMITIDA** | Clima de João Pessoa no mês t-1; conhecido no corte. |
| `casos_dengue_lag1`, `casos_chikungunya_lag1` | **PERMITIDA** | Casos do mês anterior; fechados no corte. |
| `temp*`, `umid*` (mês t) | **VAZAMENTO** | Clima do próprio mês t **não** está disponível ao decidir no fim de t-1 (previsão real). Usado só como teto rotulado. |
| `casos_dengue`, `casos_chikungunya` (mês t) | **VAZAMENTO** | Casos do mês t só se consolidam dentro/depois de t. |

**Nota sobre o clima contemporâneo:** o brief pergunta se o clima do próprio mês t pode
ser considerado conhecido. Para uma previsão **real** feita no fim de t-1, **não** — o mês
t ainda não ocorreu. Por isso o modelo honesto usa só `*_lag1`; a versão contemporânea
entra apenas como teto rotulado VAZAMENTO (e, mesmo como teto, não melhora — ver seção 5).

---

## 4. Frente 1 — features derivadas internas (evidências)

### 4.1 Correlação com o alvo (item × mês) — saída de `correlation_table`

| feature | Pearson | Spearman | classe |
|---|---|---|---|
| `media_item_ate_tm1` | 0,932 | 0,943 | PERMITIDA |
| `rollmean3_lag1` | 0,925 | 0,937 | PERMITIDA |
| `rollmean2_lag1` | 0,912 | 0,933 | PERMITIDA |
| `qtd_total_lag1` | 0,880 | 0,920 | PERMITIDA |
| `doses_dia_item_lag1` | 0,548 | 0,508 | PERMITIDA |
| `tendencia_local` | −0,012 | 0,033 | PERMITIDA |
| `hosp_ativos_lag1` | 0,041 | 0,022 | PERMITIDA |
| `hosp_diversidade_lag1` | −0,015 | −0,008 | PERMITIDA |
| `mes_do_ano`, `trimestre` | ≈0 | ≤0,01 | PERMITIDA |

Leitura: as únicas features derivadas fortemente correlacionadas são **todas reexpressões
do nível/histórico do item** (média expansível, médias móveis, lag-1). Elas são
redundantes entre si (multicolinearidade alta, como nas investigações anteriores). As
features **novas de verdade** — pressão hospitalar defasada, diversidade, tendência local,
calendário — têm correlação **próxima de zero**. O perfil posológico (`doses_dia_item_lag1`)
tem correlação moderada (0,51), mas é em grande parte um atributo **estável do item** (cada
antimicrobiano tem posologia típica), ou seja, mais um proxy de "qual item é" do que um
sinal temporal.

### 4.2 Importância por RandomForest (só ranking exploratório)

`media_item_ate_tm1` (0,49) + `rollmean3_lag1` (0,27) + `rollmean2_lag1` (0,11) =
**~87% da importância concentrada no nível e na suavização da própria demanda**. Pressão
hospitalar, diversidade, calendário e **todas as externas** ficam com ≤0,008 cada.

### 4.3 R² incremental (forward selection, linear, in-train)

| passo | feature | R² acumulado | ΔR² |
|---|---|---|---|
| 1 | `media_item_ate_tm1` | 0,841 | 0,841 |
| 2 | `rollmean2_lag1` | 0,843 | 0,002 |
| 3 | `rollmean3_lag1` | 0,849 | 0,006 |
| 4 | `hosp_ativos_lag1` | 0,851 | 0,0016 |
| 5–18 | demais (posologia, externas, calendário, trend) | 0,854 (teto) | ≤0,0008 cada |

A primeira feature (média do item) entrega R²=0,84 em treino; **tudo o mais somado agrega
~0,013 de R²**. Nenhuma feature derivada interna ou externa acrescenta sinal honesto
relevante.

### 4.4 Avaliação temporal — item × mês, 20 densos (baseline WAPE 22,91%)

Ordenado por WAPE (saída de `eval_item_month`, subconjunto denso):

| modelo | features | WAPE | MAPE médio |
|---|---|---|---|
| GradientBoosting | media+pressão+posologia | **21,61%** | 63,59% |
| LinearRegression | só `media_item_ate_tm1` | 22,48% | 53,29% |
| LinearRegression | media+persistência | 22,69% | 52,91% |
| — | **baseline trivial (média do item)** | **22,91%** | 56,59% |
| LinearRegression | media+pressão+posologia | 23,21% | 60,86% |
| LinearRegression | +externas CONTEMPORÂNEAS (teto/VAZAMENTO) | 23,42% | 66,30% |
| RandomForest | derivadas internas completo | 23,48% | 65,94% |
| RandomForest | derivadas+externas lag1 (honesto completo) | 23,67% | 67,96% |
| LinearRegression | media+rolling+trend+calendário (+interação setor) | 24,23% | 49,54% |
| … | demais (externas lag1, trees com conjunto completo) | 25–38% | piora |

**Interpretação honesta:** o 21,61% do GradientBoosting não é vitória — o **mesmo conjunto
de features sob regressão linear dá 23,21%** (pior que o baseline). Quando a mesma
informação ajuda num estimador e atrapalha noutro, numa janela de 4 meses, isso é **ruído
amostral**, não sinal. Observação interessante: `media+rolling+trend+cal` reduz o **MAPE
médio** para 49,5% (melhor equilíbrio entre itens pequenos) mas piora o **WAPE** (24,2%) —
troca que não interessa ao demandante, cuja decisão é dominada pelos itens de grande volume
(onde WAPE manda). Até o **teto com vazamento contemporâneo externo** (23,42%) não supera o
baseline, confirmando que as externas não carregam sinal no nível do item.

Nos **32 itens** (baseline WAPE 24,92%) o padrão se repete: melhor honesto ≈ baseline
(24,73% da média do item), externas e derivadas não ajudam.

---

## 5. Frente 2 — dados externos (evidências do cruzamento)

### 5.1 Correlação no nível agregado total mensal (n=19) — saída de `correlation_table`

| feature | Pearson | Spearman |
|---|---|---|
| `casos_chikungunya` (t) | −0,694 | −0,651 |
| `umidmed` (t) | 0,297 | 0,377 |
| `casos_dengue` (t) | −0,473 | −0,293 |
| `tempmed` (t) | −0,338 | −0,260 |
| `umidmed_lag1` | 0,171 | 0,240 |
| `tempmed_lag1` | −0,058 | 0,187 |
| `casos_dengue_lag1` | −0,089 | −0,009 |

No **agregado** há correlações aparentes (chikungunya negativo, umidade positiva), mas:
(a) com **n=19** qualquer |r|<~0,46 não é significativo a 5%; (b) as correlações mais
fortes são **contemporâneas** (mês t = VAZAMENTO) e **somem ao defasar** para t-1
(`casos_dengue_lag1` ≈ 0; `tempmed_lag1` ≈ 0,19). Ou seja, mesmo a correlação bruta é em
boa parte coincidência de duas séries sazonais curtas, não relação defasada utilizável.

No nível **item × mês** (seção 4.1) as externas têm |Spearman| ≤ 0,03 — **sem relação**.

### 5.2 Avaliação temporal — total mensal agregado (baseline MAPE 9,55% / WAPE 9,82%)

Janela fixa de 4 meses (saída de `eval_aggregate`):

| modelo | features | MAPE | WAPE |
|---|---|---|---|
| LinearRegression | **media+clima_lag1** | **7,89%** | **7,31%** |
| LinearRegression | clima CONTEMPORÂNEO (teto/VAZAMENTO) | 8,13% | 8,64% |
| RandomForest | media+clima_lag1 | 8,19% | 8,31% |
| LinearRegression | clima_lag1 | 8,34% | 8,44% |
| LinearRegression | só média | 8,95% | 8,89% |
| — | **baseline trivial (média)** | **9,55%** | 9,82% |
| LinearRegression | epidemia_lag1 | 9,60% | 8,97% |
| — | baseline persistência | 15,85% | 15,04% |

Na janela fixa, `media+clima_lag1` parece vencer por ~1,7 pp de MAPE. **Mas o teste tem só
4 pontos.**

### 5.3 Teste de robustez — backtest de origem expansível (saída de `backtest_aggregate`)

Para checar se o ganho é real, repetimos a previsão **um-passo-à-frente** movendo a origem
(10 previsões sucessivas, o uso realista mês a mês):

| modelo | n origens | MAPE | WAPE |
|---|---|---|---|
| baseline média (expanding) | 10 | **10,21%** | 10,45% |
| media+clima_lag1 (expanding) | 10 | **14,74%** | 15,03% |

**O ganho desaparece e vira perda.** O clima defasado, quando testado fora de uma janela
de 4 meses escolhida, **piora** a previsão em ~4,5 pp de MAPE. A conclusão honesta é
inequívoca: a vitória da seção 5.2 é **sobreajuste à janela de teste**, não sinal. Isso é
coerente com n=19, 6 features climáticas e correlação defasada ≈ 0.

---

## 6. Insights descritivos úteis ao demandante (mesmo sem virar preditor)

Mesmo sem ganho preditivo, o cruzamento com dados públicos rende leituras úteis:

1. **João Pessoa tem sazonalidade climática clara** (figura `fe4_sazonalidade_clima.png`):
   período mais ameno e úmido em jun–ago (temp média ~24,5 °C, umidade ~83%) e mais quente
   e seco em nov–mar (~26,8 °C, ~73%). **A demanda de antimicrobianos do HU, porém, não
   acompanha esse ciclo** — ela é um processo estável de nível alto (figuras
   `fe1_demanda_vs_clima.png` e `fe2_demanda_vs_epidemia.png`), reforçando que o hospital
   atende um fluxo de internações relativamente constante, não dirigido por sazonalidade
   climática/arboviral local.
2. **As arboviroses locais (dengue/chikungunya) não são motor da demanda de antimicrobianos**,
   o que faz sentido clínico: são viroses, não indicam antibioticoterapia. A fonte que
   **poderia** importar — SRAG respiratória (InfoGripe) — não baixou; é o próximo candidato
   a testar, não as arboviroses.
3. **O melhor sinal honesto é interno e de nível:** a demanda do próximo mês de cada item é
   essencialmente seu próprio patamar histórico (média/médias móveis). Operacionalmente,
   isso **valida** a prática de reposição por consumo médio e indica que esforço de coleta
   extra renderia mais em **granularidade/consistência do registro** do que em variáveis
   externas.

---

## 7. Veredito final para a Entrega 5

**Melhor modelo HONESTO de regressão (número exato):**

- **Nível item × mês (20 densos), unidade recomendada ao demandante:** regressão linear
  sobre `media_item_ate_tm1` → **WAPE ≈ 22,5%**, que **empata** com o baseline trivial por
  item (**WAPE 22,91% / MAPE médio 56,6%**, `modeling/baseline_by_item.py`). Nenhuma feature
  derivada interna nem externa supera isso de forma consistente.
- **Nível total mensal agregado:** a **média histórica (MAPE 9,55%)** permanece o melhor
  modelo honesto e robusto. O `media+clima_lag1` só "ganha" na janela fixa (MAPE 7,89%) e
  **perde no backtest** (14,74%), logo **não** deve ser adotado.

**Conclusão:** as três investigações convergem. A demanda de antimicrobianos do HU-UFPB é
um **processo estável de nível, sem estrutura temporal nem resposta a clima/epidemia local
exploráveis de forma honesta** nesta janela de 19 meses. O modelo honesto **empata** com o
baseline — um resultado legítimo pela rubrica ("se o modelo não supera o baseline, relate
isso"). Recomenda-se reportar o empate com transparência e usar os dados externos como
**insight descritivo**, não como preditor.

### Recomendações de próximos passos (se houver tempo/dados)

1. **Tentar InfoGripe/SRAG novamente** (fonte respiratória, a mais plausível para
   antibióticos) — endpoint do GitLab esteve instável; vale novo mirror/retry ou o CSV do
   OpenDataSUS SRAG filtrado por PB.
2. **Mais histórico** (>24 meses) é o que realmente destravaria sazonalidade; a conclusão
   "sem sinal" é válida para esta janela e revisável com mais dados.
3. Manter o baseline por item como preditor de produção; não adotar features externas.

---

## 8. Reprodutibilidade e artefatos

- **Scripts** (adicionados, nenhum módulo existente substituído):
  - `src/huforecast/analysis/external_sources.py` — download real (InfoDengue), tentativas
    documentadas (INMET, InfoGripe), cache em `data/external/`. Rodar:
    `python -m huforecast.analysis.external_sources`.
  - `src/huforecast/analysis/derived_external_investigation.py` — features derivadas +
    externas, correlação/importância/R²/avaliação temporal + backtest. Rodar:
    `python -m huforecast.analysis.derived_external_investigation`.
  - `src/huforecast/config.py` — adicionada apenas a constante `EXTERNAL_DIR`
    (aditiva, não quebra nada).
- **Dados externos** (`data/external/`, público agregado mensal, pode ser versionado):
  `joaopessoa_clima_epidemia_mensal.csv` e `FONTES.json`. **Nenhum dado do hospital foi
  gravado fora de `data/raw/` nem versionado.**
- **Figuras** (`docs/figuras/`): `fe1_demanda_vs_clima.png`, `fe2_demanda_vs_epidemia.png`,
  `fe3_corr_derivadas_externas.png`, `fe4_sazonalidade_clima.png`.
- **Determinismo:** `SEED = 42`; tabelas de avaliação idênticas em execuções repetidas
  (verificado). Pipeline de produção (`import huforecast; hf.run_pipeline()`) confirmado
  funcionando após as adições.

*Conteúdo redigido pela equipe; nenhum número foi inventado — todos vêm da execução real
sobre `data/raw/` e sobre os dados públicos efetivamente baixados da InfoDengue.*


---

## 9. Segunda rodada de fontes externas (busca exaustiva) — adendo

Após a primeira rodada (clima + arboviroses via InfoDengue), esgotamos as fontes
externas plausíveis, atrás especialmente do sinal **respiratório** (SRAG), que é o
mais plausível clinicamente para antibioticoterapia (pneumonia bacteriana secundária
a vírus respiratórios). Resultado de todas as tentativas, com diagnóstico de
conectividade real:

| Fonte tentada | Rota | Status | Observação |
|---|---|---|---|
| **InfoGripe / SRAG (Fiocruz)** | `gitlab.procc.fiocruz.br` | **FALHOU** | TCP recusado (http=000). DNS resolve (157.86.153.131), mas o datacenter não aceita conexão. |
| **InfoGripe / SRAG (site)** | `info.gripe.fiocruz.br/data/detailed/...` | **FALHOU** | Mesmo datacenter Fiocruz (157.86.x), TCP recusado. |
| **InfoGripe / SRAG (mirror)** | `github.com/infogripe/Boletim_InfoGripe` | **OK** | **Mirror oficial no GitHub, vivo e atualizado diariamente.** Série de SRAG por UF; filtramos PB. |
| **PAMEpi (Fiocruz Rondônia)** | `pamepi.rondonia.fiocruz.br` | **FALHOU** | TCP recusado (Fiocruz). |
| **OpenDataSUS SRAG** | `opendatasus.saude.gov.br` (CKAN) | **FALHOU** | API CKAN desativada; portal virou SPA. Microdados anuais (centenas de MB) inviáveis aqui. |
| **API Dados Abertos MS** | `apidadosabertos.saude.gov.br` | **PARCIAL** | Responde rápido, mas só arboviroses (sem respiratório) e sem filtro de UF confiável. |
| **INMET (estação A320)** | `apitempo.inmet.gov.br` | **FALHOU** | Corpo vazio sem token. |

**Diagnóstico honesto:** o datacenter da Fiocruz (`157.86.x.x`) está inacessível a
partir daqui — não é erro de código nem timeout ajustável; os domínios resolvem no DNS
mas recusam a conexão TCP. Contornamos pelo **mirror oficial do InfoGripe no GitHub**
(`infogripe/Boletim_InfoGripe`), de onde baixamos a série de SRAG da Paraíba de verdade.
Tudo está materializado em `src/huforecast/analysis/external_sources.py`
(`fetch_infogripe_srag`), com a tentativa ao GitLab registrada como falha documentada
em `data/external/FONTES.json`.

### 9.1 SRAG respiratória da Paraíba — série obtida (mirror GitHub)

Série mensal de casos reportados de SRAG na PB (2025-03→2026-09), com sazonalidade
respiratória clara (picos em mai–jul/2025 e mar–mai/2026):

| mês | SRAG casos | | mês | SRAG casos |
|---|---|---|---|---|
| 2025-03 | 206 | | 2026-01 | 245 |
| 2025-04 | 251 | | 2026-02 | 275 |
| 2025-05 | 585 | | 2026-03 | 526 |
| 2025-06 | 582 | | 2026-04 | 825 |
| 2025-07 | 657 | | 2026-05 | 648 |
| 2025-08 | 485 | | 2026-06 | 372 |
| 2025-09 | 422 | | 2026-07 | 342 |
| ... | ... | | 2026-08 | 281 |
| | | | 2026-09 | 250 |

### 9.2 Teste honesto do SRAG e do nível de alerta epidemiológico

Dois novos sinais externos foram testados na demanda **total mensal** (n=19), com
correlação de Spearman **defasada (lag-1, honesta)** e **backtest de origem expansível**
(10 previsões um-passo-à-frente, o uso realista mês a mês):

| sinal externo (lag-1) | Spearman lag-1 | backtest MAPE | baseline média MAPE | veredito |
|---|---|---|---|---|
| **SRAG respiratória PB** | +0,30 (n.s.) | 9,97% | 9,94% | empata / não ajuda |
| **nível de alerta (InfoDengue)** | +0,485 (limiar) | 9,75% | 9,94% | melhora marginal (~0,2 pp), robusta |
| clima (`media+clima_lag1`) | — | 14,74% | 10,21% | **piora** (sobreajuste, 1ª rodada) |

Limiar de significância aprox. para n=18: |rho| ~ 0,48 (p=0,05).

**Leitura honesta:**

1. **A SRAG respiratória — a fonte clinicamente mais plausível — não carrega sinal
   preditivo honesto** para a demanda agregada: correlação defasada fraca e não
   significativa (+0,30), e no backtest empata com o baseline (9,97% vs 9,94%).
2. **O único sinal externo que não piora** é o **nível de alerta epidemiológico**
   (`nivel`) defasado: rho +0,485 (no limiar) e backtest 9,75% vs 9,94% — melhora de
   ~0,2 pp, **robusta** (não colapsa como o clima). É um empate técnico com leve
   vantagem, não uma vitória material.
3. **Confirmação definitiva:** testadas as três famílias de sinal externo
   plausíveis — **clima, arbovirose e respiratória** — **nenhuma supera o baseline de
   forma material e robusta** nesta janela de 19 meses. O limite é o **comprimento da
   série**, não a falta de fontes (ver `docs/revisao_literatura_estado_da_arte.md`,
   §7–§8: com poucos dados, modelos simples são difíceis de bater; a série tem baixa
   *forecastability* mensurável).

### 9.3 Conclusão do adendo

Esgotada a busca externa, o veredito da seção 7 se mantém e sai **reforçado**: a melhor
estimativa honesta de demanda do próximo mês continua sendo o **nível histórico do
item** (item × mês) / a **média** (agregado). Os sinais externos entram no relatório
como **investigação rigorosa e insight descritivo**, não como preditor adotado — com a
ressalva de que `nivel_lag1` é o único candidato externo honesto com leve vantagem
robusta no agregado, disponível para trabalho futuro com mais histórico.

*Fontes externas efetivamente baixadas e materializadas em
`src/huforecast/analysis/external_sources.py`; status de cada uma em
`data/external/FONTES.json`. Nenhum número foi inventado.*
