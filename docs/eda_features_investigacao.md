# Investigação de features e formulação — EDA da Entrega 5

> Análise **somente leitura**. Nada do pacote de produção foi alterado. Todo o
> código desta investigação vive em `src/huforecast/analysis/features_investigation.py`
> e roda com `python -m huforecast.analysis.features_investigation`
> (semente fixa `SEED = 42`, sem caminhos absolutos, via `huforecast.config`).
> Todos os números abaixo vêm de execução real sobre os dados em `data/raw/`
> (`Planilha CCIH 2026 Atualizado.xlsx`, 7.261 tratamentos).

---

## Resumo executivo (o que seguir)

Testamos exaustivamente **features estruturais** ("por dentro" da demanda),
**unidades/alvos alternativos** e **vários modelos** na mesma partição temporal
(treino 2025-03→2026-05, teste 2026-06→2026-09, métrica principal WAPE).

**Conclusão central: nenhuma formulação com features honestas (sem vazamento)
supera de forma material o baseline trivial da média do item.** No escopo
estabelecido (20 antimicrobianos densos), o baseline a superar é **WAPE = 22,9%**;
o melhor modelo honesto chega a **WAPE ≈ 22,5%** (regressão linear usando só a
média histórica do item), uma diferença de ~0,4 ponto percentual que não
representa ganho real. Modelos de árvore com features estruturais defasadas
**pioram** (23,4–26%). Só quando se usam features **contemporâneas** (tratamentos
ativos/iniciados *no próprio mês t*) o erro despenca para ~5,8% WAPE — mas essas
features são **VAZAMENTO** e não estão disponíveis no momento da decisão.

**Formulação recomendada para o modelo da Entrega 5:** manter a unidade
**antimicrobiano × mês** nos 20 itens densos, com a média histórica do item
como preditor de nível, materializada como uma **regressão linear simples sobre
`media_item_ate_t-1` (+ opcionalmente `qtd_total_lag1`)**. O número a bater é
**WAPE 22,9% / MAPE médio 56,6%** (baseline trivial por item). O resultado honesto
a reportar é "o modelo empata com o baseline" — que é um resultado legítimo pela
rubrica ("se o modelo não supera o baseline, relate isso"). No nível **total
mensal agregado**, o baseline da média é ainda mais forte (MAPE 9,6% / WAPE 9,8%)
e também não há sinal honesto que o supere.

---

## 1. O que foi testado

| Frente | Pergunta | Como foi medido |
|---|---|---|
| **A** | Features estruturais "por dentro" da demanda têm poder preditivo? | Painel item × mês com 11 features estruturais, em dois tempos: contemporâneo (mês t) e defasado (mês t-1) |
| **B** | Outra unidade/alvo dá mais sinal? | item×mês (atual), item×setor×mês, total mensal, agregação semanal |
| **C** | Quais relações existem? | Pearson + Spearman (feature×alvo), matriz de correlação (multicolinearidade), importância por RandomForest (só ranking), R² incremental (forward) |
| **D** | Qual formulação recomendar? | Avaliação WAPE/MAPE de cada conjunto na partição temporal vs. baseline correspondente |

Partição e métrica idênticas às do semestre (`huforecast.config.TEST_MONTHS = 4`,
`INCOMPLETE_MONTH = "2026-10"` excluído). WAPE é a métrica principal; MAPE médio
por item é a secundária (definições em `modeling/baseline_by_item.py`, reusadas
em `analysis/features_investigation.py`).

---

## 2. Classificação de vazamento (requisito obrigatório)

Regra: para prever o mês **t**, só vale informação disponível **até o fim de t-1**.
Cada feature candidata foi classificada explicitamente
(ver dicionário `LEAKAGE_TABLE` em `features_investigation.py`).

### Features contemporâneas (mês t) — **VAZAMENTO**

| Feature | Classe | Justificativa |
|---|---|---|
| `n_ativos` | VAZAMENTO | Tratamentos ativos no mês t só se conhecem durante/após t |
| `n_iniciados` | VAZAMENTO | Inícios de t são conhecidos ao longo de t, não no corte |
| `n_encerrados` | VAZAMENTO | Encerramentos ocorrem dentro de t |
| `qtd_dia_ativos` | VAZAMENTO | Depende dos tratamentos ativos de t |
| `qtd_dia_iniciados` | VAZAMENTO | Depende dos inícios de t |
| `dur_media_iniciados` | VAZAMENTO | Duração dos tratamentos iniciados em t |
| `n_setores_ativos` | VAZAMENTO | Setores ativos em t |
| `n_setores_iniciados` | VAZAMENTO | Setores dos inícios de t |
| `frac_uti`, `frac_clinmed`, `frac_dip` | VAZAMENTO | Mix setorial dos inícios de t |

### Features defasadas (mês t-1) e nível histórico — **PERMITIDAS**

| Feature | Classe | Justificativa |
|---|---|---|
| `n_ativos_lag1` | PERMITIDA | Ativos no mês anterior; fechado no corte |
| `n_iniciados_lag1` | PERMITIDA | Inícios do mês anterior |
| `n_encerrados_lag1` | PERMITIDA | Encerramentos do mês anterior |
| `qtd_dia_ativos_lag1` | PERMITIDA | Carga diária dos ativos no mês anterior |
| `qtd_dia_iniciados_lag1` | PERMITIDA | Inícios do mês anterior |
| `dur_media_iniciados_lag1` | PERMITIDA | Duração média dos inícios do mês anterior |
| `n_setores_ativos_lag1` | PERMITIDA | Setores ativos no mês anterior |
| `n_setores_iniciados_lag1` | PERMITIDA | Setores dos inícios do mês anterior |
| `frac_uti_lag1`, `frac_clinmed_lag1`, `frac_dip_lag1` | PERMITIDA | Mix setorial do mês anterior |
| `qtd_total_lag1` | PERMITIDA | Demanda do mês anterior (persistência) |
| `media_item_ate_tm1` | PERMITIDA | Média histórica do item até t-1 (nível) |

**Ponto decisivo:** as features estruturais mais correlacionadas com a demanda
(`qtd_dia_iniciados`, `qtd_dia_ativos`, `n_iniciados`, `n_ativos`) são justamente
as **contemporâneas** — ou seja, são quase uma reexpressão da própria demanda do
mês e são **vazamento**. Suas versões defasadas, que são as utilizáveis, têm poder
muito menor (ver Frente C).

---

## 3. Frente C — relações entre features e alvo

### 3.1 Correlação com o alvo (item × mês)

Spearman de cada feature com `qtd_total` do mês t (saída de `correlation_table`):

| feature | Pearson | Spearman | classe |
|---|---|---|---|
| `qtd_dia_iniciados` | 0,981 | 0,994 | **VAZAMENTO** |
| `qtd_dia_ativos` | 0,978 | 0,984 | **VAZAMENTO** |
| `n_iniciados` | 0,690 | 0,957 | **VAZAMENTO** |
| `n_ativos` | 0,704 | 0,949 | **VAZAMENTO** |
| `media_item_ate_tm1` | 0,932 | 0,943 | PERMITIDA |
| `qtd_dia_ativos_lag1` | 0,896 | 0,931 | PERMITIDA |
| `n_encerrados` | 0,671 | 0,926 | **VAZAMENTO** |
| `qtd_dia_iniciados_lag1` | 0,886 | 0,922 | PERMITIDA |
| `n_setores_iniciados` | 0,670 | 0,922 | **VAZAMENTO** |
| `qtd_total_lag1` | 0,880 | 0,920 | PERMITIDA |
| `n_ativos_lag1` | 0,666 | 0,897 | PERMITIDA |
| `n_iniciados_lag1` | 0,647 | 0,889 | PERMITIDA |
| `n_encerrados_lag1` | 0,633 | 0,883 | PERMITIDA |
| `frac_clinmed` / `frac_dip` / `frac_uti` | 0,07–0,27 | 0,46–0,58 | VAZAMENTO |
| `dur_media_iniciados(_lag1)` | 0,29–0,31 | 0,45–0,53 | — |
| `frac_*_lag1` | 0,06–0,28 | 0,38–0,54 | PERMITIDA |

Leitura: o que mais correlaciona é contemporâneo (vazamento). Entre as PERMITIDAS,
`media_item_ate_tm1`, `qtd_dia_ativos_lag1`, `qtd_dia_iniciados_lag1` e
`qtd_total_lag1` lideram — mas todas são essencialmente proxies do **nível médio
do item**, não um sinal temporal novo. O mix setorial defasado tem correlação
fraca (≤0,54). Figura: `docs/figuras/f1_corr_features_permitidas.png`.

### 3.2 Multicolinearidade entre as PERMITIDAS

A matriz de correlação de Spearman (`feature_corr_matrix`,
`docs/figuras/f2_matriz_correlacao.png`) mostra forte redundância entre
`media_item_ate_tm1`, `qtd_total_lag1`, `qtd_dia_ativos_lag1` e
`qtd_dia_iniciados_lag1` — todas medem o mesmo "tamanho" do item. Isso explica por
que adicionar várias delas quase não acrescenta informação (ver R² incremental).

### 3.3 Importância por RandomForest (somente ranking exploratório)

Treino nos meses de treino, só features PERMITIDAS (`tree_importance`):

| feature | importância |
|---|---|
| `media_item_ate_tm1` | 0,726 |
| `qtd_dia_ativos_lag1` | 0,076 |
| `qtd_dia_iniciados_lag1` | 0,052 |
| `qtd_total_lag1` | 0,050 |
| `dur_media_iniciados_lag1` | 0,020 |
| `frac_dip_lag1` | 0,016 |
| `n_encerrados_lag1` | 0,015 |
| demais (`frac_*`, `n_*_lag1`, setores) | ≤ 0,012 cada |

**73% da importância está na média histórica do item.** As features estruturais
defasadas são marginais.

### 3.4 R² incremental (forward selection, linear, in-train)

| passo | feature adicionada | R² acumulado | ΔR² |
|---|---|---|---|
| 1 | `media_item_ate_tm1` | 0,852 | 0,852 |
| 2 | `qtd_dia_ativos_lag1` | 0,854 | 0,002 |
| 3 | `n_encerrados_lag1` | 0,858 | 0,004 |
| 4 | `qtd_total_lag1` | 0,866 | 0,007 |
| 5 | `n_ativos_lag1` | 0,869 | 0,004 |
| 6 | `qtd_dia_iniciados_lag1` | 0,873 | 0,004 |
| 7–13 | demais | 0,876 (teto) | ≤ 0,002 cada |

A primeira feature (média do item) já entrega R²=0,85 em treino; **todas as 12
features estruturais defasadas somadas adicionam ~0,023 de R²**. Isso é o retrato
de "não há sinal estrutural honesto além do nível do item". (R² alto em treino não
se traduz em ganho no teste — ver Frente D.)

---

## 4. Frente B — unidades/alvos alternativos

| Formulação | Baseline trivial (WAPE) | Baseline persistência (WAPE) | Observação |
|---|---|---|---|
| **item × mês** (20 densos, atual) | **22,9%** | 31,4% | Escopo e número estabelecidos do semestre |
| item × mês (todos 32 itens) | 24,9% | 33,8% | Itens esparsos pioram tudo |
| **item × setor × mês** | 50,3% | 66,4% | 210 células, **53,6% são zero** → muito esparso, inviável |
| **total mensal agregado** | 9,8% (MAPE 9,6%) | 15,0% (MAPE 15,8%) | Baseline mais forte; menos útil ao demandante (decide item a item) |
| **semanal agregado** | — | — | 82 semanas; autocorr lag-1 = **-0,11**, lag-4 = -0,08 |

Leituras:

- **Mais granular (item×setor×mês) é pior**, não melhor: pulveriza o volume e gera
  mais de metade de células vazias. Descartado.
- **Agregar no total mensal** tem o baseline mais difícil de bater (média 9,6%
  MAPE), mas não atende a decisão da farmácia (reposição item a item) e não
  apresenta sinal temporal explorável.
- **Semanal não ajuda:** dar mais pontos (82 semanas) não cria memória — a
  autocorrelação de lag-1 é **negativa e fraca** (-0,11), confirmando ruído branco
  em torno de um nível estável (CV ≈ 0,24). Mais pontos de um processo sem memória
  não geram sinal temporal.

Conclusão da Frente B: **item × mês (20 densos) continua sendo a melhor unidade**
para o demandante, e nenhuma unidade alternativa revela sinal honesto superável.

---

## 5. Frente D — avaliação na partição (comparação honesta)

### 5.1 Itens densos (20) — escopo oficial, baseline 22,9% WAPE

Ordenado por WAPE (saída de `temporal_eval_item_month`, subconjunto denso):

| modelo | features | WAPE | MAPE médio |
|---|---|---|---|
| RandomForest | **VAZAMENTO contemporâneo (teto)** | **5,79%** | 8,82% |
| GradientBoosting | VAZAMENTO contemporâneo (teto) | 5,92% | 12,20% |
| LinearRegression | VAZAMENTO contemporâneo (teto) | 8,90% | 25,48% |
| **LinearRegression** | **só `media_item_ate_tm1`** | **22,48%** | 53,29% |
| — | **baseline trivial (média do item)** | **22,91%** | 56,59% |
| RandomForest | PERMITIDAS + média item | 23,44% | 68,29% |
| RandomForest | PERMITIDAS completo | 23,51% | 69,05% |
| LinearRegression | PERMITIDAS completo | 24,17% | 58,83% |
| GradientBoosting | PERMITIDAS completo | 25,94% | 70,20% |
| RandomForest | só estruturais PERMITIDAS | 26,02% | 90,87% |
| — | baseline persistência | 31,43% | 75,65% |
| LinearRegression | só estruturais PERMITIDAS | 33,30% | 82,75% |

### 5.2 Todos os 32 itens — baseline 24,9% WAPE

Mesmo padrão: teto com vazamento = **6,43% WAPE** (RandomForest); melhor honesto =
LinearRegression só média do item = 24,73% (baseline 24,92%); árvores com
PERMITIDAS = 25,6–28%.

### 5.3 Interpretação

1. **O sinal preditivo real existe, mas é vazamento.** Com features
   contemporâneas o WAPE cai de ~23% para ~6% — prova de que a demanda do mês é
   quase determinada pelos tratamentos ativos/iniciados *naquele mês*. Só que esses
   números não são conhecidos no momento da decisão (fim de t-1).
2. **Com features honestas, o teto prático é o próprio baseline.** A regressão
   linear sobre a média histórica do item apenas **empata** com o baseline trivial
   (22,5% vs 22,9%), diferença dentro do ruído de 4 meses de teste. As features
   estruturais defasadas e o mix setorial **não agregam** e com árvores chegam a
   **piorar** (overfitting no nível do item sem ganho de sinal).
3. **Persistência é claramente pior** (31,4%), coerente com a autocorrelação
   lag-1 ~0 já conhecida: repetir o mês anterior é inferior a usar a média.

---

## 6. Recomendação final

**Formulação a seguir na Entrega 5:**

- **Alvo:** demanda mensal planejada `qtd_total`.
- **Unidade:** antimicrobiano × mês, restrita aos **20 itens densos** (presentes em
  todos os meses) — a curva ABC concentra o volume neles e é onde a decisão de
  reposição importa.
- **Features (todas PERMITIDAS):** `media_item_ate_tm1` como preditor de nível,
  opcionalmente `qtd_total_lag1`. Modelo: **regressão linear** (interpretável, sem
  overfit). Evitar árvores com o conjunto estrutural completo — pioram no teste.
- **Número a superar:** **WAPE 22,9% / MAPE médio 56,6%** (baseline trivial por
  item, `modeling/baseline_by_item.py`). No total mensal, o número é MAPE 9,6%.

**Resultado honesto esperado e como reportá-lo:** o modelo **empata** com o
baseline (não o supera de forma material). Pela rubrica da disciplina, isso é um
resultado válido e deve ser relatado como tal ("se o modelo não supera o baseline,
relate isso — é um resultado, não um fracasso"). A cadeia de raciocínio fica sem
buracos: medimos tendência/sazonalidade/memória (ausentes em 19 meses), medimos
features estruturais (fortes mas vazamento), e medimos as versões honestas
(sem ganho). A demanda de antimicrobianos do HU é um **processo estável de nível
alto, sem estrutura temporal explorável honestamente** — a melhor estimativa
honesta para o próximo mês é a média histórica do item.

### Por que nenhuma disposição supera o baseline (síntese causal)

- A demanda mensal de um item é a soma de tratamentos cujo início/duração só se
  revelam **dentro** do mês. O que prediz bem (ativos/iniciados em t) é
  contemporâneo à própria demanda → vazamento.
- Defasar para t-1 remove o vazamento mas também remove o sinal, porque **não há
  memória**: autocorrelação lag-1 ~0 (mensal) e -0,11 (semanal). O passado próximo
  não informa o presente além do nível médio.
- Logo, o melhor preditor honesto colapsa no **nível médio do item**, que é
  exatamente o baseline trivial.

---

## 7. Limitações e notas de reprodutibilidade

- **Janela curta:** 19 meses (15 de treino) é pouco para caracterizar sazonalidade
  anual; a conclusão "sem sinal temporal" é válida para esta janela e pode ser
  revista com mais histórico.
- **Teste de 4 meses:** diferenças de ~0,4 pp em WAPE estão dentro do ruído
  amostral; por isso "empate" é a leitura correta, não "vitória marginal".
- **Normalização de setor** agrupa em UTI / Clínica Médica / DIP / Outro; variações
  de grafia (`UTI ADULTO` vs `UTI - ADULTO`) são absorvidas pelo prefixo.
- **RandomForest/GradientBoosting** usados **apenas para ranking exploratório** e
  como referência; a recomendação de produção é linear.
- **Reprodutibilidade:** `SEED = 42` fixado; execução determinística confirmada
  (RF PERMITIDAS_completo denso = 23,5108% em execuções repetidas). Rodar com
  `python -m huforecast.analysis.features_investigation`.

### Artefatos gerados

- Script: `src/huforecast/analysis/features_investigation.py` (+ `__init__.py`).
- Figuras em `docs/figuras/`:
  - `f1_corr_features_permitidas.png` — Spearman das features permitidas vs alvo.
  - `f2_matriz_correlacao.png` — multicolinearidade entre permitidas.
  - `f3_dispersao_feature_mais_forte.png` — dispersão da feature permitida mais forte vs demanda.
  - `f4_series_top_itens.png` — séries mensais dos 5 itens de maior volume, com corte treino/teste.

*Conteúdo redigido pela equipe; nenhum número foi inventado — todos vêm da execução
real sobre `data/raw/`.*
