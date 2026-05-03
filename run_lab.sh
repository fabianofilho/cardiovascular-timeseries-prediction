#!/usr/bin/env bash
# run_lab.sh - Pipeline completo para rodar no computador do laboratorio
# Uso: bash run_lab.sh [--skip-extract] [--models sarima,prophet,timesfm,xgboost,catboost]
set -euo pipefail

# -----------------------------------------------------------------------
# Parametros configuráveis
# -----------------------------------------------------------------------
UF="SP"
YEARS="2010-2023"          # Serie estendida (vs. 2019-2023 original)
HORIZON=6
MIN_TRAIN=36               # 3 anos minimos com serie estendida
MODELS="sarima,prophet,timesfm"
SKIP_EXTRACT=false
VENV_DIR=".venv"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# -----------------------------------------------------------------------
# Flags de linha de comando
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-extract)   SKIP_EXTRACT=true; shift ;;
    --models)         MODELS="$2"; shift 2 ;;
    --years)          YEARS="$2"; shift 2 ;;
    --horizon)        HORIZON="$2"; shift 2 ;;
    --min-train)      MIN_TRAIN="$2"; shift 2 ;;
    *) echo "[ERRO] Flag desconhecida: $1"; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------
# Diretorios e logs
# -----------------------------------------------------------------------
mkdir -p "$LOG_DIR" data/raw data/processed results
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================="
echo "  cardiovascular-timeseries-prediction - run_lab.sh"
echo "  Data: $(date)"
echo "  Anos: $YEARS | Modelos: $MODELS | Horizon: $HORIZON"
echo "======================================================="

# -----------------------------------------------------------------------
# Ambiente virtual
# -----------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Criando ambiente virtual em $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[INFO] Python: $(which python) - $(python --version)"

# -----------------------------------------------------------------------
# Dependencias base (sempre)
# -----------------------------------------------------------------------
echo "[INFO] Instalando dependencias base ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install -e . --quiet

# Instalar Prophet se solicitado
if echo "$MODELS" | grep -q "prophet"; then
  echo "[INFO] Instalando Prophet ..."
  pip install --quiet "prophet>=1.1.5"
fi

# Instalar TimesFM se solicitado
if echo "$MODELS" | grep -q "timesfm"; then
  echo "[INFO] Instalando TimesFM (pode demorar na primeira vez) ..."
  pip install --quiet "torch>=2.5.0"
  pip install --quiet "timesfm @ git+https://github.com/google-research/timesfm.git" || \
    echo "[WARN] TimesFM falhou - continuando sem ele"
fi

# Instalar XGBoost/CatBoost se solicitado
if echo "$MODELS" | grep -qE "xgboost|catboost"; then
  echo "[INFO] Instalando gradient boosting ..."
  pip install --quiet xgboost catboost scikit-learn skforecast
fi

# -----------------------------------------------------------------------
# Extracao de dados (SIM/DataSUS)
# -----------------------------------------------------------------------
RAW_CSV="data/raw/sim_real_${UF}_${YEARS//-/_}.csv"
SERIES_CSV="data/processed/serie_cv_${UF}_${YEARS//-/_}.csv"
META_JSON="data/processed/meta_${UF}_${YEARS//-/_}.json"
REPORT_JSON="results/data_reality_report_${TIMESTAMP}.json"

if [ "$SKIP_EXTRACT" = false ]; then
  echo ""
  echo "[ETAPA 1/4] Extraindo dados SIM ($UF, $YEARS) ..."
  echo "[AVISO] Download do DataSUS pode levar 30-60 min para 14 anos de dados."
  python scripts/extract_sim_real.py \
    --uf "$UF" \
    --years "$YEARS" \
    --max-retries 5 \
    --retry-wait 30 \
    --raw-output "$RAW_CSV" \
    --series-output "$SERIES_CSV" \
    --meta-output "$META_JSON"
  echo "[OK] Extracao concluida."
else
  echo ""
  echo "[ETAPA 1/4] --skip-extract ativo. Buscando serie processada existente ..."
  # Fallback: usar a mais recente que existir
  if [ ! -f "$SERIES_CSV" ]; then
    SERIES_CSV=$(ls data/processed/serie_*.csv 2>/dev/null | tail -1 || true)
    if [ -z "$SERIES_CSV" ]; then
      echo "[ERRO] Nenhuma serie processada encontrada. Remova --skip-extract."
      exit 1
    fi
    echo "[INFO] Usando: $SERIES_CSV"
  fi
fi

# -----------------------------------------------------------------------
# Validacao dos dados
# -----------------------------------------------------------------------
echo ""
echo "[ETAPA 2/4] Validando dataset ..."
python scripts/validate_real_dataset.py \
  --raw-csv   "$RAW_CSV" \
  --series-csv "$SERIES_CSV" \
  --meta-json  "$META_JSON" \
  --report-output "$REPORT_JSON" || {
    echo "[WARN] Validacao retornou avisos. Verifique $REPORT_JSON antes de usar os resultados."
  }
echo "[OK] Validacao concluida. Relatorio: $REPORT_JSON"

# -----------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------
OUTPUT_PREFIX="results/benchmark_${UF}_${YEARS//-/_}_${TIMESTAMP}"

echo ""
echo "[ETAPA 3/4] Rodando benchmark ($MODELS) ..."
python scripts/run_benchmark.py \
  --input-csv    "$SERIES_CSV" \
  --date-col     date \
  --value-col    value \
  --freq         MS \
  --horizon      "$HORIZON" \
  --min-train-size "$MIN_TRAIN" \
  --models       "$MODELS" \
  --output-prefix "$OUTPUT_PREFIX"
echo "[OK] Benchmark concluido."

# -----------------------------------------------------------------------
# Figuras
# -----------------------------------------------------------------------
echo ""
echo "[ETAPA 4/4] Gerando figuras ..."
python scripts/generate_paper_figures.py \
  --metrics-csv "${OUTPUT_PREFIX}_metrics.csv" \
  --predictions-csv "${OUTPUT_PREFIX}_predictions.csv" \
  --series-csv "$SERIES_CSV" \
  --output-dir "images/lab_${TIMESTAMP}" || \
    echo "[WARN] generate_paper_figures.py falhou ou requer ajuste de parametros."

# -----------------------------------------------------------------------
# Resumo final
# -----------------------------------------------------------------------
echo ""
echo "======================================================="
echo "  CONCLUIDO: $(date)"
echo "  Log completo: $LOG_FILE"
echo "  Metricas: ${OUTPUT_PREFIX}_metrics.csv"
echo "  Previsoes: ${OUTPUT_PREFIX}_predictions.csv"
echo "======================================================="

echo ""
echo "[RESULTADO] Top modelos por sMAPE:"
python - <<'PYEOF'
import sys, os, glob
prefix = sorted(glob.glob("results/benchmark_*_metrics.csv"))
if not prefix:
    print("  Nenhum CSV de metricas encontrado.")
    sys.exit(0)
latest = prefix[-1]
import csv
rows = list(csv.DictReader(open(latest)))
rows.sort(key=lambda r: float(r.get("smape", 999)))
for r in rows:
    print(f"  {r['model']:20s}  MAE={float(r['mae']):8.1f}  RMSE={float(r['rmse']):8.1f}  sMAPE={float(r['smape']):.2f}%")
PYEOF
