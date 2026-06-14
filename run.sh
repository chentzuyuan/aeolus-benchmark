#!/bin/bash
# run.sh — One-click runner: GBDT → DL (separate processes, nohup)
# Usage: bash run.sh
# Re-runnable: already-done models are skipped via checkpoint.json

set -e
PYTHON=/path/to/School_Projects/_thesis_env/bin/python3
DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS="$DIR/logs"
mkdir -p "$LOGS"

echo "=========================================="
echo "Aeolus Benchmark — Full Run"
echo "  Step 1: GBDT (RF/XGB/CatBoost/XGB-Optuna)"
echo "  Step 2: DL   (MLP/ResNet/AutoInt)"
echo "  Checkpoint: $DIR/checkpoint.json"
echo "=========================================="

# Step 1: GBDT (runs in this shell, logs to file + stdout)
echo ""
echo "[Step 1/2] Running GBDT..."
"$PYTHON" "$DIR/gbdt.py" 2>&1 | tee "$LOGS/gbdt.log"

echo ""
echo "[Step 2/2] Running DL (separate process — nohup)..."
nohup "$PYTHON" "$DIR/dl.py" > "$LOGS/dl.log" 2>&1 &
DL_PID=$!
echo "  DL running as PID=$DL_PID"
echo "  Tail log: tail -f $LOGS/dl.log"
echo ""

# Wait for DL to finish
wait $DL_PID
echo ""
echo "=========================================="
echo "ALL DONE — see checkpoint.json for results"
echo "=========================================="
