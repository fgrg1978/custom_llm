#!/usr/bin/env bash
# Evaluacion standalone: evalua phase1_model.pt y phase3_model.pt contra Stockfish,
# elige el ganador y escribe RESULT.txt. NO entrena nada.
#
# Usar cuando el training ya termino/se paro y solo queremos la evaluacion.
# Robusto: no usa `set -e`, un fallo de evaluate no aborta el script.

set -uo pipefail  # OJO: sin -e a proposito

cd ~/custom_llm
source venv/bin/activate

THRESHOLD_ELO="${THRESHOLD_ELO:-600}"
CKPT_DIR=domains/chess/checkpoints

echo "=========================================================="
echo "  EVALUACION STANDALONE"
echo "  Quality gate: >= $THRESHOLD_ELO ELO"
echo "=========================================================="

# best_model.pt actual = resultado de Phase 3 (lo dejo el training parado).
# Guardarlo como phase3_model.pt si no existe ya.
if [ ! -f "$CKPT_DIR/phase3_model.pt" ]; then
    if [ -f "$CKPT_DIR/best_model.pt" ]; then
        cp "$CKPT_DIR/best_model.pt" "$CKPT_DIR/phase3_model.pt"
        echo "Phase 3 guardado: phase3_model.pt (desde best_model.pt)"
    fi
fi

if [ ! -f "$CKPT_DIR/phase1_model.pt" ]; then
    echo "ERROR: no existe phase1_model.pt"
    exit 1
fi

eval_model() {
    local label=$1
    local path=$2
    if [ ! -f "$path" ]; then
        echo "0"
        return
    fi
    cp "$path" "$CKPT_DIR/best_model.pt"
    echo "" >&2
    echo "Evaluando $label ($path)..." >&2
    python cli.py chess evaluate --games 30 --levels 0 1 2 3 4 --seed 42 \
        > /tmp/eval_${label}.log 2>&1
    cat /tmp/eval_${label}.log >&2
    local elo
    elo=$(grep -oE "(Estimacion puntual|Cota inferior|Cota superior): ~-?[0-9]+" /tmp/eval_${label}.log | head -1 \
        | grep -oE "\-?[0-9]+$")
    echo "${elo:-0}"
}

PHASE1_ELO=$(eval_model "phase1" "$CKPT_DIR/phase1_model.pt")
echo "Phase 1 ELO: $PHASE1_ELO"

PHASE3_ELO=$(eval_model "phase3" "$CKPT_DIR/phase3_model.pt")
echo "Phase 3 ELO: $PHASE3_ELO"

# Winner
if [ "${PHASE3_ELO:-0}" -gt "${PHASE1_ELO:-0}" ] 2>/dev/null; then
    WINNER="phase3"; WINNER_ELO=$PHASE3_ELO
else
    WINNER="phase1"; WINNER_ELO=$PHASE1_ELO
fi
cp "$CKPT_DIR/${WINNER}_model.pt" "$CKPT_DIR/best_model.pt"

if [ "${WINNER_ELO:-0}" -ge "$THRESHOLD_ELO" ] 2>/dev/null; then
    STATUS="ACEPTADO"
else
    STATUS="RECHAZADO - no supera threshold $THRESHOLD_ELO"
fi

cat > /tmp/RESULT.txt <<EOF
Status:       $STATUS
Winner:       $WINNER
Winner ELO:   $WINNER_ELO
Phase 1 ELO:  $PHASE1_ELO  (epoch 27, val_loss 1.8751)
Phase 3 ELO:  $PHASE3_ELO  (epoch 9, val_loss 1.8800)
Threshold:    $THRESHOLD_ELO
Date:         $(date)
EOF

echo ""
echo "=========================================================="
echo "  $STATUS  -  winner: $WINNER (~$WINNER_ELO ELO)"
echo "=========================================================="
cat /tmp/RESULT.txt
