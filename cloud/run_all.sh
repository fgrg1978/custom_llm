#!/usr/bin/env bash
# Pipeline completo: setup -> distill+phase1 paralelo -> phase3 -> evaluate.
# Maximiza uso de la instancia: GPU para training, CPUs para Stockfish distill simultaneo.
#
# Target: A100 40GB SXM4 (30 vCPUs). Coste estimado: ~6-8h, ~$12-16 a $1.99/h.

set -euo pipefail

cd ~/custom_llm
THRESHOLD_ELO="${THRESHOLD_ELO:-600}"

echo "=========================================================="
echo "  PIPELINE COMPLETO"
echo "  Target: superar baseline Phase 0 (~420 ELO)"
echo "  Quality gate: >= $THRESHOLD_ELO ELO"
echo "=========================================================="

# ============================================================
# 1. SETUP (deps, datos, Stockfish)
# ============================================================
echo ""
echo ">>> [1/5] SETUP"
bash cloud/setup.sh
source venv/bin/activate

# ============================================================
# 2. DISTILL en background + PHASE 1 training en GPU
#    Stockfish es CPU-bound, training es GPU-bound: no compiten.
# ============================================================
echo ""
echo ">>> [2/5] DISTILL (CPU, background) + PHASE 1 TRAINING (GPU)"

# Calcular workers reservando CPUs para el DataLoader
TOTAL_CPU=$(nproc)
DISTILL_WORKERS=$(( TOTAL_CPU - 8 ))
[ $DISTILL_WORKERS -lt 4 ] && DISTILL_WORKERS=4
echo "Total CPUs: $TOTAL_CPU. Distill workers: $DISTILL_WORKERS (deja 8 para training/sistema)"

# Lanza distill en background
nohup python cli.py chess distill \
    --max-games 200000 \
    --positions-per-game 5 \
    --depth 10 \
    --workers $DISTILL_WORKERS \
    > /tmp/distill.log 2>&1 &
DISTILL_PID=$!
echo "Distill arrancado, PID: $DISTILL_PID"
sleep 5  # deja arrancar antes de que GPU consuma CPUs

# Phase 1 training (foreground, GPU)
echo ""
echo "Phase 1 training (GPU)..."
bash cloud/train.sh

# Backup Phase 1 checkpoint
cp domains/chess/checkpoints/best_model.pt domains/chess/checkpoints/phase1_model.pt
echo "Phase 1 guardado: phase1_model.pt"

# ============================================================
# 3. Esperar a que distill termine si todavia corre
# ============================================================
echo ""
echo ">>> [3/5] ESPERANDO DISTILL"
wait $DISTILL_PID || echo "Distill termino (exit code en log)"
DISTILL_OK=0
if [ -f domains/chess/data/sequences_distill.pt ]; then
    DISTILL_N=$(python -c "import torch; print(len(torch.load('domains/chess/data/sequences_distill.pt', weights_only=False)))")
    echo "Distill OK: $DISTILL_N secuencias generadas"
    DISTILL_OK=1
else
    echo "WARN: distill fallo. Saltando Phase 3."
    cat /tmp/distill.log | tail -20
fi

# ============================================================
# 4. PHASE 3 training: imitacion + distillation
# ============================================================
if [ $DISTILL_OK -eq 1 ]; then
    echo ""
    echo ">>> [4/5] PHASE 3 TRAINING (imitacion + Stockfish)"

    python cli.py chess train \
        --epochs 30 --patience 6 \
        --batch-size 384 \
        --d-model 384 --n-heads 8 --n-layers 8 \
        --max-len 192 \
        --lr 4e-4 \
        --warmup-steps 1500 \
        --use-distill \
        2>&1 | tee /tmp/train_phase3.log

    cp domains/chess/checkpoints/best_model.pt domains/chess/checkpoints/phase3_model.pt
    echo "Phase 3 guardado: phase3_model.pt"
else
    echo ""
    echo ">>> [4/5] PHASE 3 SALTADA (distill fallo)"
fi

# ============================================================
# 5. EVALUAR ambos checkpoints y elegir winner
# ============================================================
echo ""
echo ">>> [5/5] EVALUACION"

# eval_model: evalua un checkpoint. Log completo va a stderr (visible en terminal),
# SOLO el numero de ELO va a stdout (para capturar con $(...)).
eval_model() {
    local label=$1
    local path=$2
    cp "$path" domains/chess/checkpoints/best_model.pt
    echo "" >&2
    echo "Evaluando $label..." >&2
    # || true: si evaluate crashea, no matamos el script (los checkpoints ya estan a salvo)
    python cli.py chess evaluate --games 30 --levels 0 1 2 3 4 --seed 42 \
        > /tmp/eval_${label}.log 2>&1 || echo "WARN: evaluate $label fallo, ver /tmp/eval_${label}.log" >&2
    cat /tmp/eval_${label}.log >&2
    local elo
    elo=$(grep -oE "(Estimacion puntual|Cota inferior|Cota superior): ~-?[0-9]+" /tmp/eval_${label}.log | head -1 \
        | grep -oE "\-?[0-9]+$" || true)
    echo "${elo:-0}"
}

PHASE1_ELO=$(eval_model "phase1" "domains/chess/checkpoints/phase1_model.pt")
echo "Phase 1 ELO: $PHASE1_ELO"

if [ $DISTILL_OK -eq 1 ]; then
    PHASE3_ELO=$(eval_model "phase3" "domains/chess/checkpoints/phase3_model.pt")
    echo "Phase 3 ELO: $PHASE3_ELO"
else
    PHASE3_ELO=0
fi

# ============================================================
# Quality gate y winner
# ============================================================
echo ""
echo "=========================================================="
echo "  RESULTADOS"
echo "=========================================================="
echo "Phase 1 (imitacion):              ~$PHASE1_ELO ELO"
[ $DISTILL_OK -eq 1 ] && echo "Phase 3 (imitacion + Stockfish):  ~$PHASE3_ELO ELO"

if [ "$PHASE3_ELO" -gt "$PHASE1_ELO" ]; then
    WINNER="phase3"
    WINNER_ELO=$PHASE3_ELO
else
    WINNER="phase1"
    WINNER_ELO=$PHASE1_ELO
fi

cp "domains/chess/checkpoints/${WINNER}_model.pt" domains/chess/checkpoints/best_model.pt
echo ""
echo "Winner: $WINNER (~$WINNER_ELO ELO)"

if [ "$WINNER_ELO" -ge "$THRESHOLD_ELO" ]; then
    STATUS="ACEPTADO"
    EXIT_CODE=0
else
    STATUS="RECHAZADO - no supera threshold $THRESHOLD_ELO"
    EXIT_CODE=1
fi

cat > /tmp/RESULT.txt <<EOF
Status:       $STATUS
Winner:       $WINNER
Winner ELO:   $WINNER_ELO
Phase 1 ELO:  $PHASE1_ELO
Phase 3 ELO:  $PHASE3_ELO
Threshold:    $THRESHOLD_ELO
Date:         $(date)
EOF

echo ""
echo "$STATUS"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Para descargar (desde tu Mac):"
    echo "  ./cloud/download.sh ubuntu@HOST"
    echo "Y TERMINA la instancia en lambdalabs.com"
fi

exit $EXIT_CODE
