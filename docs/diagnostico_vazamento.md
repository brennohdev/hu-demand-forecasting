# Diagnóstico de vazamento (data leakage)

Corte da decisão: a previsão é feita **no fechamento do mês**, para estimar a demanda
do **mês seguinte**. Uma variável só pode entrar no modelo se já existir e estiver
estável nesse instante do corte. Classificamos abaixo as variáveis do projeto.

| variável | permitida? | justificativa em relação ao corte |
|---|---|---|
| `principio_ativo` | **Sim** | característica estável do item; conhecida antes do corte. |
| `setor` | **Sim** | conhecida no momento da prescrição; existe antes do corte. |
| `mes_inicio` / mês do ano | **Sim** | deriva da data de início; captura sazonalidade sem usar o futuro. |
| histórico de `qtd_total` dos meses anteriores | **Sim** | só usa meses já fechados (lags); base legítima da previsão. |
| `data_termino` | **Não — vazamento temporal** | só é conhecida depois que o tratamento termina; no corte, o término futuro ainda não ocorreu. |
| `situacao` (ENCERRADO/SUSPENSO) | **Não — vazamento de alvo/consequência** | é o desfecho do tratamento; só existe após o fim. Usá-la para prever demanda futura é olhar o resultado. |
| `qtd_dispensada` / `qtd_resta` | **Não — consequência da decisão** | registradas conforme o consumo acontece; aparecem só depois da decisão de dispensar. |

## As três formas de vazamento, aplicadas ao projeto
- **De alvo:** usar `situacao` ou `qtd_total` do próprio mês-alvo como entrada.
- **Temporal:** usar `data_termino` ou qualquer agregado que inclua meses ainda não fechados.
- **De partição:** ajustar imputação/escalonamento com dados de teste, ou deixar o mesmo
  mês aparecer em treino e teste. Evitado com partição temporal estrita (treino = meses
  antigos; teste = últimos meses) e `fit` só no treino.

## Cuidado especial (consequência da decisão)
Variáveis como `qtd_dispensada`, `qtd_resta` e `situacao` parecem ótimas preditoras
porque correlacionam fortemente com a demanda — mas só existem **porque o tratamento já
aconteceu**. Em produção, no momento da previsão, elas ainda não existiriam. Por isso
são proibidas.
