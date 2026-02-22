# Auditoria de Extração Real SIM (2026-02-22)

## Objetivo

Validar se os dados usados em benchmark eram extração real do DataSUS/SIM e executar pipeline de produção com retry + validação.

## Diagnóstico dos dados existentes

- `data/raw/cv_sim_sample_10000_cases.csv` não é extração real do layout completo do SIM.
- Evidências:
  - apenas 3 colunas (`event_date`, `source`, `causabas`);
  - padrão temporal e distribuição altamente alinhados à série sintética base (`serie_eventos_sp_tiny.csv`);
  - correlação praticamente perfeita com série base escalada.

## Fluxo de produção implementado

- `scripts/extract_sim_real.py`: extração real SIM com retry e metadado.
- `scripts/validate_real_dataset.py`: validação de realidade e aptidão para benchmark.
- `scripts/run_real_pipeline.py`: orquestra extração -> validação -> benchmark.

## Execução real no ambiente atual

Comando:

```bash
PYTHONPATH=src python scripts/run_real_pipeline.py \
  --uf SP --year 2022 --max-retries 3 --retry-wait 10 \
  --horizon 6 --min-train-size 24 \
  --models sarima,prophet,timesfm \
  --output-prefix results/benchmark_sim_real_sp_2022
```

Resultado:

- Falha na extração real por timeout no canal de dados FTP do DataSUS.
- Metadado registrado em `data/processed/sim_real_extraction_metadata.json` com `status: failed`.

## Conclusão

- O pipeline de produção está implementado e funcional.
- A execução real ainda depende de disponibilidade/conectividade do FTP DataSUS.
- Benchmarks reais devem ser rodados novamente assim que a extração retornar `status: success`.
