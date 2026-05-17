#!/usr/bin/env bash
# Continuacion manual del pipeline cuando se para Phase 1 antes de tiempo.
# Hace los pasos [4/5] y [5/5] de run_all.sh: Phase 3 + evaluacion + quality gate.
#
# Usar cuando: se mato Phase 1 manualmente (y con el run_all.sh) y queremos
# seguir con Phase 3 sobre el checkpoint de Phase 1 ya guardado.
#
# Prerequisitos:
#   - domains/chess/checkpoints/best_model.pt  = resultado de Phase 1
#   - domains/chess/data/sequences_distill.pt  = output del distill
#   - domains/chess/data/sequences.pt + vocab.json

set -euo pipefail

cd ~/custom_llm
source venv/bin/activate

THRESHOLD_ELO="${THRESHOLD_ELO:-600}"

echo "=========================================================="
echo "  CONTINUACION: PHASE 3 + EVALUACION"
echo "  Quality gate: >= $THRESHOLD_ELO ELO"
echo "=========================================================="

# Guardar el checkpoint de Phase 1 (lo que dejo el training parado)
if [ ! -f domains/chess/checkpoints/best_model.pt ]; then
    echo "ERROR: no existe best_model.pt (resultado de Phase 1). Abortando."
    exit 1
fi
cp domains/chess/checkpoints/best_model.pt domains/chess/checkpoints/phase1_model.pt
echo "Phase 1 guardado: phase1_model.pt"

# Verificar distill
DISTILL_OK=0
if [ -f domains/chess/data/sequences_distill.pt ]; then
    DISTILL_N=$(python -c "import torch; print(len(torch.load('domains/chess/data/sequences_distill.pt', weights_only=False)))")
    echo "Distill disponible: $DISTILL_N secuencias"
    DISTILL_OK=1
else
    echo "WARN: no hay sequences_distill.pt -> Phase 3 se salta, solo se evalua Phase 1"
fi

# ============================================================
# PHASE 3: imitacion + distillation
# ============================================================
if [ $DISTILL_OK -eq 1 ]; then
    echo ""
    echo ">>> PHASE 3 TRAINING (imitacion + Stockfish)"
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
fi

# ============================================================
# EVALUACION
# ============================================================
echo ""
echo ">>> EVALUACION"

eval_model() {
    local label=$1
    local path=$2
    cp "$path" domains/chess/checkpoints/best_model.pt
    echo "" >&2
    echo "Evaluando $label..." >&2
    python cli.py chess evaluate --games 30 --levels 0 1 2 3 4 --seed 42 \
        > /tmp/eval_${label}.log 2>&1 || echo "WARN: evaluate $label fallo" >&2
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
# Quality gate + winner
# ============================================================
if [ "$PHASE3_ELO" -gt "$PHASE1_ELO" ]; then
    WINNER="phase3"; WINNER_ELO=$PHASE3_ELO
else
    WINNER="phase1"; WINNER_ELO=$PHASE1_ELO
fi
cp "domains/chess/checkpoints/${WINNER}_model.pt" domains/chess/checkpoints/best_model.pt

if [ "$WINNER_ELO" -ge "$THRESHOLD_ELO" ]; then
    STATUS="ACEPTADO"; EXIT_CODE=0
else
    STATUS="RECHAZADO - no supera threshold $THRESHOLD_ELO"; EXIT_CODE=1
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
echo "=========================================================="
echo "  $STATUS  -  winner: $WINNER (~$WINNER_ELO ELO)"
echo "=========================================================="
cat /tmp/RESULT.txt

exit $EXIT_CODE
