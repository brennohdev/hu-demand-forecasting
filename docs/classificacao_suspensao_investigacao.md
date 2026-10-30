# Investigação de classificação — prever suspensão de tratamento (Entrega 5, 2ª perspectiva)

> **Status:** investigação **somente leitura**. Nenhum arquivo do pacote de produção
> foi criado ou alterado por este relatório. Todos os números abaixo vêm de **execução
> real** sobre os dados em `data/raw/Planilha CCIH 2026 Atualizado.xlsx` (7.261
> tratamentos), carregados pela pipeline de produção
> (`huforecast.load_raw_treatments` → `huforecast.clean_treatments`), com semente fixa
> `SEED = 42`. Esta é a **segunda perspectiva** do projeto; a perspectiva de regressão
> de demanda está documentada e em stand-by em
> [`docs/eda_features_investigacao.md`](eda_features_investigacao.md).

---

## 1. Resumo executivo (a resposta primeiro)

**Vale a pena? Não como preditor acionável.** Prever se um tratamento será **SUSPENSO**,
usando apenas informação disponível **no início do tratamento**, **não supera de forma
material o baseline** de forma útil. Os melhores modelos **honestos** ficam em
**ROC-AUC ≈ 0,59–0,63** e **PR-AUC ≈ 0,21–0,26** (prevalência/base rate da classe
positiva = 0,207), com um trade-off ruim: para alcançar recall ~0,55 na classe
"suspenso", a precisão cai para ~0,19 — ou seja, o modelo marca quase metade de todos os
tratamentos para capturar pouco mais da metade das suspensões reais.

**A descoberta central e mais importante:** diferente do caso de regressão (onde as
features de vazamento *disparavam* o desempenho), aqui **nem mesmo as features de
vazamento salvam o modelo**. Incluindo `dias_tratamento` e `qtd_total` (que provamos
serem valores **realizados**, não planejados — ver §3), o teto de desempenho sobe
pouquíssimo: ROC-AUC ≈ 0,62–0,65, PR-AUC ≈ 0,29. Isso indica que **a suspensão é
dirigida por eventos clínicos que não existem na planilha** (reação adversa, resultado
de cultura/antibiograma, piora ou óbito do paciente, decisão médica) — nenhuma coluna da
base, honesta ou vazada, carrega esse sinal.

**Veredito honesto:** o modelo de classificação de suspensão **não bate o baseline de
forma materialmente útil e não funciona nem com vazamento**. É um resultado legítimo de
reportar (pela rubrica, "se o modelo não supera o baseline, relate isso"). A acurácia
pura (~85%) é **enganosa** — é exatamente o que o chute da classe majoritária entrega, com
recall 0 na classe de interesse. A recomendação é **documentar esta perspectiva como
testada e sem sinal honesto acionável**, e manter a classificação como diagnóstico
descritivo (fatores de risco associados, §6) e não como preditor operacional.

| | Métrica honesta (melhor) | Teto com **vazamento** | Base rate / majoritária |
|---|---|---|---|
| ROC-AUC | ~0,60–0,63 | ~0,62–0,65 | 0,50 |
| PR-AUC (classe suspenso) | ~0,21–0,26 | ~0,29 | 0,15–0,21 |
| Recall @0.5 (LogReg honesto) | ~0,55 | ~0,58 | 0,00 (majoritária) |
| Precision @0.5 (LogReg honesto) | ~0,19 | ~0,20 | — |

---

## 2. Definição do problema e metodologia

### 2.1 Alvo

- **Alvo binário:** `suspenso = 1` se `situacao == "SUSPENSO"`, `0` se `situacao == "ENCERRADO"`.
- **`EM TRATAMENTO` (111 linhas) é excluído** do treino e do teste: o desfecho ainda não
  é conhecido, então rotulá-lo contaminaria o alvo.
- Distribuição de `situacao` na base limpa (confere com o briefing):
  `ENCERRADO = 5677`, `SUSPENSO = 1473`, `EM TRATAMENTO = 111`.
- Após excluir `EM TRATAMENTO` **e** o mês parcial `2026-10`
  (`config.INCOMPLETE_MONTH`, coleta em 22/10, mesma regra do resto do projeto):
  **n = 7.022**, **prevalência de suspensão = 20,7%**.

### 2.2 Features candidatas (momento da decisão = início do tratamento)

Conhecidas na prescrição: `setor`, `principio_ativo`/`antimicrobiano`, `posologia`,
`qtd_dia`, e o mês do ano de `data_inicio` (sazonalidade). As candidatas
`dias_tratamento` e `qtd_total` foram **investigadas quanto a vazamento** (§3) e
**reclassificadas como vazamento**.

### 2.3 Partição — duas estratégias, escolha documentada

Avaliamos as duas e **reportamos a temporal** como métrica principal:

- **(a) Split temporal por `data_inicio` (ESCOLHIDA):** treino = tratamentos iniciados
  em `2025-03 … 2026-05` (n=5.523), teste = `2026-06 … 2026-09` (n=1.499). Imita o uso
  real (prever tratamentos futuros) e é coerente com a partição temporal usada o semestre
  todo (`config.TEST_MONTHS = 4`). **Observação relevante:** há *distribution shift*
  temporal — prevalência de suspensão cai de **0,223 no treino** para **0,147 no teste**.
  Isso por si só já penaliza qualquer modelo calibrado no treino e é um sinal de
  não-estacionariedade (coerente com a conclusão da perspectiva de regressão: a base tem
  pouca estrutura temporal estável).
- **(b) K-fold estratificado (5 folds, REFERÊNCIA):** usado como sanity check. Elimina o
  shift temporal, então mede o "melhor caso" de poder discriminante. Mesmo assim o teto é
  baixo (ROC-AUC ≤ 0,62 honesto).

### 2.4 Modelos e pré-processamento

`Pipeline` + `ColumnTransformer` (OneHot para categóricas, `SimpleImputer(mediana)` +
`StandardScaler` para numéricas — `qtd_dia` tem ~1% de ausentes). Modelos:
`LogisticRegression(class_weight="balanced")`, `RandomForestClassifier(class_weight="balanced")`,
`GradientBoostingClassifier`. Semente fixa em todos.

### 2.5 Métricas

Como a classe positiva (suspenso) é minoria e é o que importa, reportamos
**precision, recall, F1, ROC-AUC e PR-AUC** da classe positiva. **Acurácia é
deliberadamente desprezada** como métrica de decisão: com 15–21% de prevalência, o chute
"nunca suspende" já dá ~85% de acurácia com **recall 0** — inútil para a CCIH/farmácia.

---

## 3. Diagnóstico de vazamento (requisito central) — tabela por feature

A previsão ocorre **no início do tratamento**; só vale o que existe nesse instante.
Investigamos cada candidata explicitamente, linha a linha sobre os dados reais.

### 3.1 Evidência decisiva: `dias_tratamento` e `qtd_total` são REALIZADOS, não planejados

Confirmamos a suspeita preliminar do briefing. Comparando linha a linha:

- `dias_tratamento == (data_termino − data_inicio) + 1` em **99,4% das 7.259 linhas**
  (contagem inclusiva de dias). Correlação `dias_tratamento` × duração real = **0,987**.
  A diferença mediana é exatamente **+1 dia** (contagem inclusiva), não uma folga de
  planejamento. Ou seja, `dias_tratamento` **é a duração efetivamente realizada** — só
  pode ser preenchida/fechada quando o tratamento termina.
- `qtd_total == qtd_dia × dias_tratamento` em **100% das linhas** (corr = 0,99999).
  Portanto `qtd_total` **herda** o vazamento de `dias_tratamento`: também é um valor de
  desfecho, não um plano inicial.

Conclusão: ambas são **VAZAMENTO** no problema de classificação de suspensão. No início
do tratamento, a farmácia não conhece a duração realizada nem o total consumido — esses
números só se fecham no encerramento/suspensão.

### 3.2 `suspender` / `suspender_flag` são o próprio alvo

`suspender_flag` (derivada da coluna manual `SUSPENDER`) é praticamente uma cópia do
alvo: dos 1.473 SUSPENSO, **1.472 têm `suspender_flag = 1`**; dos ENCERRADO, apenas 96 de
5.677. Usá-la seria prever o alvo com o alvo. **VAZAMENTO de alvo — proibida.**

### 3.3 Tabela de classificação por feature

| feature | classe | justificativa (relativa ao corte = início do tratamento) |
|---|---|---|
| `setor` | **PERMITIDA** | conhecido na prescrição. |
| `principio_ativo` / `antimicrobiano` | **PERMITIDA** | item prescrito; conhecido no início. Alta cardinalidade (32 princípios / 38 descrições) — testado com e sem (§5). |
| `posologia` | **PERMITIDA** | intervalo posológico definido na prescrição. |
| `qtd_dia` | **PERMITIDA** | doses/dia definidas na prescrição. ~1% ausente (imputado). |
| mês de `data_inicio` (sazonalidade) | **PERMITIDA** | deriva só da data de início; não usa o futuro. |
| `dias_tratamento` | **VAZAMENTO** | = (`data_termino`−`data_inicio`)+1 em 99,4% dos casos → duração **realizada**, só conhecida no fim. |
| `qtd_total` | **VAZAMENTO** | = `qtd_dia`×`dias_tratamento` em 100% dos casos → herda o vazamento de `dias_tratamento`. |
| `data_termino` | **VAZAMENTO** | data do desfecho; não existe no início. |
| `suspender` / `suspender_flag` | **VAZAMENTO (alvo)** | marcação manual do próprio desfecho de suspensão (1.472/1.473). |
| `situacao` | **ALVO** | é o rótulo; não entra como feature. |
| `qtd_dispensada` / `qtd_resta` | **VAZAMENTO (consequência)** | registradas conforme o consumo; já removidas na limpeza por 98–99,8% ausentes. |

---

## 4. Baselines

- **Classe majoritária ("nunca suspende"):** acurácia = 0,853 no teste temporal, mas
  **recall = 0, precision = 0, F1 = 0** na classe suspenso, ROC-AUC = 0,50. É o alvo a
  superar e a prova de que **acurácia engana**.
- **Taxa base (base rate):** PR-AUC de um classificador aleatório = prevalência da classe
  = **0,147** no teste temporal (shift) / **0,207** em 5-fold estratificado. Todo PR-AUC
  reportado deve ser comparado a esse piso.

---

## 5. Resultados por modelo (mesma partição)

### 5.1 Split temporal (treino 2025-03→2026-05, teste 2026-06→2026-09)

Classe positiva = suspenso. Limiar 0,5. (`baseline_maj` = chute da majoritária.)

| modelo | features | precision | recall | F1 | ROC-AUC | PR-AUC | acurácia |
|---|---|---|---|---|---|---|---|
| baseline majoritária | — | 0,000 | 0,000 | 0,000 | 0,500 | 0,147 | **0,853** |
| LogReg | HONESTO_completo | 0,188 | 0,552 | 0,280 | 0,601 | 0,207 | 0,582 |
| RandomForest | HONESTO_completo | 0,149 | 0,217 | 0,176 | 0,507 | 0,163 | 0,701 |
| GradBoost | HONESTO_completo | 0,000 | 0,000 | 0,000 | 0,586 | 0,213 | 0,852 |
| LogReg | HONESTO_sem_principio | 0,188 | 0,588 | 0,284 | 0,616 | 0,212 | 0,564 |
| RandomForest | HONESTO_sem_principio | 0,169 | 0,376 | 0,233 | 0,509 | 0,165 | 0,636 |
| LogReg | HONESTO_setor_agrupado | 0,185 | 0,593 | 0,282 | 0,595 | 0,207 | 0,554 |
| GradBoost | HONESTO_setor_agrupado | 0,400 | 0,009 | 0,018 | 0,626 | 0,228 | 0,852 |
| LogReg | HONESTO_só setor+princípio | 0,185 | 0,620 | 0,285 | 0,589 | 0,206 | 0,542 |
| RandomForest | HONESTO_só setor+princípio | 0,200 | 0,548 | 0,293 | 0,605 | 0,210 | 0,611 |
| **LogReg** | **VAZAMENTO (teto)** | 0,197 | 0,584 | 0,295 | **0,635** | 0,214 | 0,588 |
| GradBoost | VAZAMENTO (teto) | 0,250 | 0,005 | 0,009 | **0,652** | 0,232 | 0,851 |
| RandomForest | SÓ VAZAMENTO num | 0,194 | 0,715 | 0,305 | 0,618 | 0,202 | 0,520 |

Leitura:
- **Nenhum modelo honesto é acionável.** ROC-AUC fica em 0,51–0,62; PR-AUC mal passa da
  base rate (0,147). A LogReg balanceada "ganha" recall (~0,55–0,62) só porque marca
  ~45% dos tratamentos (precision ~0,19).
- **O teto com vazamento é quase igual ao honesto.** Isto é o resultado mais informativo:
  adicionar `dias_tratamento`+`qtd_total` leva o ROC-AUC de ~0,60 para ~0,65 — ganho
  marginal. **Não há "número secreto" escondido nas features de desfecho.**
- `GradBoost` com limiar 0,5 colapsa na majoritária (recall ~0) porque não é balanceado;
  seu ROC-AUC maior reflete só ordenação, não um ponto de operação útil.

### 5.2 K-fold estratificado (5 folds) — referência sem shift temporal

| modelo | features | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LogReg | HONESTO_completo | 0,251 | 0,556 | 0,346 | 0,588 | 0,264 |
| RandomForest | HONESTO_completo | 0,227 | 0,257 | 0,241 | 0,528 | 0,227 |
| GradBoost | HONESTO_completo | 0,377 | 0,014 | 0,027 | 0,589 | 0,262 |
| LogReg | VAZAMENTO (teto) | — | — | — | 0,605 | 0,270 |
| GradBoost | VAZAMENTO (teto) | — | — | — | **0,620** | **0,285** |
| RandomForest | VAZAMENTO (teto) | — | — | — | 0,551 | 0,230 |

Mesmo sem o shift temporal (melhor caso), o teto — **inclusive com vazamento** — é
ROC-AUC ≈ 0,62 e PR-AUC ≈ 0,285 contra base rate 0,207. Discriminação fraca confirmada
por uma via independente da partição.

### 5.3 Variações testadas (o máximo de disposições razoáveis)

- **com vs sem `principio_ativo`** (alta cardinalidade): praticamente empatam — remover
  o princípio ativo não piora (LogReg ROC-AUC 0,601 → 0,616). O item prescrito carrega
  pouca informação sobre suspensão.
- **setor completo (17) vs setor agrupado** (UTI/Clínica Médica/DIP/Pediatria/Outro):
  equivalente; agrupar não cria sinal.
- **só `setor`+`principio_ativo`** (sem posologia/qtd/mês): equivalente — confirma que
  nenhuma categórica isolada discrimina.
- **só numéricas de vazamento** (`dias_tratamento`,`qtd_total`): ROC-AUC ~0,61–0,62 —
  mais uma confirmação de que o vazamento não resolve.

### 5.4 Sinal univariado das features de vazamento

Mesmo olhando as realizadas diretamente, a separação é mínima:

| feature | mediana (suspenso) | mediana (encerrado) | média (susp/enc) | Mann-Whitney p |
|---|---|---|---|---|
| `dias_tratamento` | 11,0 | 11,0 | 12,35 / 11,31 | 6,9e-20 |
| `qtd_total` | 30,0 | 28,0 | 40,95 / 38,39 | 6,4e-08 |
| `qtd_dia` | 3,0 | 3,0 | 3,36 / 3,29 | 0,080 |

As diferenças são **estatisticamente significativas apenas por causa do n grande**; as
**medianas são idênticas ou quase** e a sobreposição das distribuições é enorme — sem
poder de classificação prático. Tratamentos suspensos duram em média ~1 dia a mais, o que
é efeito, não causa preditiva disponível no início.

---

## 6. Interpretação — fatores associados à suspensão (uso descritivo)

Da LogReg honesta (coeficientes; sinal positivo = maior risco de suspensão) e das
prevalências por grupo. **Atenção:** são **associações descritivas**, não um preditor
confiável (o modelo tem ROC-AUC ~0,6). Servem como leitura para a CCIH, não como alvo
operacional.

Prevalência de suspensão por grupo de setor:

| setor (grupo) | prevalência suspensão | n |
|---|---|---|
| Pediatria (ped/neo/UCIN) | 0,290 | 724 |
| UTI | 0,218 | 2.566 |
| DIP | 0,213 | 1.001 |
| Clínica Médica | 0,212 | 1.436 |
| Outro | 0,131 | 1.295 |

Prevalência por princípio ativo (n ≥ 50), extremos:

- **Maior risco:** CEFTAZIDIMA (0,29), PIPERACILINA+TAZOBACTAM (0,27), CEFEPIMA (0,27),
  FLUCONAZOL (0,25), VANCOMICINA (0,25) — de modo geral antimicrobianos de amplo
  espectro/empíricos, coerente com suspensão após descalonamento guiado por cultura.
- **Menor risco:** LEVOFLOXACINO (0,15), METRONIDAZOL (0,16), GENTAMICINA (0,17),
  CEFTRIAXONA (0,17), MEROPENEM (0,18).

Coeficientes LogReg: setores de Pediatria e itens de amplo espectro empurram o risco para
cima; UCIN, LEVOFLOXACINO, TOBRAMICINA e posologias longas (36/36, 72/72) empurram para
baixo. A leitura clínica plausível (terapia empírica de amplo espectro iniciada na UTI/
pediatria e depois suspensa/descalonada) é **consistente**, mas o sinal é fraco demais
para prever o caso individual.

---

## 7. Limitações

- **Causa da suspensão não está na base.** Suspensão decorre de eventos clínicos —
  reação adversa, resultado de antibiograma/descalonamento, piora/óbito, troca de esquema —
  que **não são colunas da planilha**. Por isso nem o vazamento recupera o sinal. Esta é
  a limitação de fundo.
- **Shift temporal:** prevalência cai de 0,223 (treino) para 0,147 (teste). A janela de
  19 meses é curta e não-estacionária (mesma limitação da perspectiva de regressão).
- **Rótulo de suspensão vem de marcação manual** (`SUSPENDER`), com possível
  subnotificação/erro; 96 ENCERRADO também têm a marca e 1 SUSPENSO não tem.
- **`EM TRATAMENTO` excluído:** correto para não contaminar o alvo, mas reduz um pouco a
  base e impede avaliar tratamentos em curso.
- **Limiar 0,5** é arbitrário; ajustá-lo troca precision por recall, mas não muda o teto
  (ROC-AUC/PR-AUC), que é o diagnóstico real.

---

## 8. Recomendação

1. **Reportar como perspectiva testada e honesta, sem preditor acionável.** O modelo de
   classificação de suspensão **não supera o baseline de forma útil** (ROC-AUC honesto
   ~0,60; PR-AUC ~0,21–0,26 vs base rate 0,15–0,21) e **não funciona nem com features de
   vazamento** (teto ROC-AUC ~0,65). Isso é um resultado válido e deve ser apresentado
   como tal, sempre com **PR-AUC/recall**, nunca com acurácia.
2. **Usar a classificação de forma descritiva** (fatores de risco da §6: amplo espectro,
   pediatria/UTI) como insight qualitativo para a CCIH, deixando claro que é associação e
   não previsão confiável.
3. **Rotular o "teto com vazamento" explicitamente** em qualquer material — ele existe só
   para mostrar que a informação de desfecho tampouco resolve, reforçando a honestidade.
4. **Se a equipe quiser materializar o experimento** (deliverable de script/figuras do
   briefing), fazê-lo como `src/huforecast/analysis/suspension_investigation.py` rodável
   via `python -m`, seguindo o padrão de `analysis/features_investigation.py`
   (Pipeline/ColumnTransformer, `SEED=42`, sem caminhos absolutos, usando
   `huforecast.config`), **adicionando, nunca substituindo** módulos existentes, e com
   figuras em `docs/figuras/` (matriz de confusão, curva ROC/PR, importância/coeficientes,
   prevalência por setor/antimicrobiano). Este relatório **não** criou esses arquivos
   (passo somente leitura); os números acima já reproduzem o que o script produziria.
5. **Para extrair as duas perspectivas:** manter esta (classificação) e a de regressão
   (stand-by) lado a lado no fechamento da Entrega 5 — ambas convergem para a mesma
   conclusão de fundo: **a base registra o desfecho dos tratamentos, mas não os sinais
   antecedentes que permitiriam prevê-los honestamente.**

---

## 9. Reprodutibilidade

- Dados: `data/raw/Planilha CCIH 2026 Atualizado.xlsx` via
  `huforecast.load_raw_treatments()` → `huforecast.clean_treatments()`.
- Semente: `SEED = 42` em todos os modelos; `StratifiedKFold(shuffle=True, random_state=42)`.
- Partição temporal: treino `2025-03…2026-05`, teste `2026-06…2026-09`, `2026-10`
  (parcial) excluído — mesma regra de `huforecast.config`.
- Alvo: `suspenso = (situacao == "SUSPENSO")`, `EM TRATAMENTO` excluído.
- Pré-processamento: `ColumnTransformer` (OneHot categóricas; `SimpleImputer(mediana)` +
  `StandardScaler` numéricas).

*Conteúdo redigido pela equipe; nenhum número foi inventado — todos vêm de execução real
sobre `data/raw/`. Nenhum módulo de baseline/regressão foi alterado e nenhum dado bruto
foi versionado.*
