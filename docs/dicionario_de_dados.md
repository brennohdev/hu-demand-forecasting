# Dicionário de dados — base de antimicrobianos (HU-UFPB)

Fonte bruta: `data/raw/Planilha CCIH 2026 Atualizado.xlsx`, aba `Planilha`
("PLANILHA DE TRATAMENTO DE PACIENTES COM ANTIMICROBIANO").
Base processada: `data/processed/tratamentos_antimicrobianos.parquet`.

Unidade de análise: **um tratamento** de um paciente com um antimicrobiano.
Período coberto: 2025-03 a 2026-10 (o mês 2026-10 é incompleto, coleta em 22/10).

| coluna (processada) | origem (planilha) | tipo | descrição |
|---|---|---|---|
| `setor` | SETOR | categórico | setor do hospital (UTI adulto, clínica médica, DIP, etc.) |
| `situacao` | SITUAÇÃO | categórico | estado do tratamento: ENCERRADO, SUSPENSO, EM TRATAMENTO |
| `dias_tratamento` | DIAS | numérico | duração prevista do tratamento, em dias |
| `posologia` | POSO | texto | intervalo posológico (ex.: 12/12, 8/8) |
| `qtd_dia` | QTD/DIA | numérico | doses/ampolas por dia |
| `qtd_total` | TOTAL | numérico | quantidade total prevista do tratamento (usada como demanda) |
| `suspender` | SUSPENDER | texto | marca manual quando o tratamento é suspenso (maioria vazia) |
| `data_inicio` | INÍCIO | data | início do tratamento |
| `data_termino` | TÉRMINO | data | término (previsto/registrado) do tratamento |
| `antimicrobiano` | ANTIMICROBIANO | categórico | descrição completa do item (princípio ativo + dose + forma) |
| `qtd_dispensada` | QTD | numérico | quantidade dispensada (quase sempre vazia — descartada) |
| `principio_ativo` | derivado | categórico | primeiro termo do `antimicrobiano` (ex.: MEROPENEM) |
| `mes_inicio` | derivado | texto (AAAA-MM) | mês de início, para a série temporal |
| `suspender_flag` | derivado | binário | 1 se o tratamento foi marcado para suspender, 0 caso contrário |

## Colunas removidas na limpeza
- `qtd_resta` (99.8% ausente) e `qtd_dispensada` (98.3% ausente): sem informação útil.

## Observações de qualidade
- Nome do paciente: **já vem anonimizado (vazio)** na planilha recebida. Não há dado pessoal identificável na base.
- 164 linhas idênticas duplicadas foram removidas.
- O mês 2026-10 está incompleto (coleta em 22/10) e é excluído da modelagem.
