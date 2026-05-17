#!/usr/bin/env bash
# Descarga el checkpoint y vocab del H100 al local.
# Uso: SSH_KEY=~/.ssh/lambda.pem ./cloud/download.sh ubuntu@LAMBDA_IP
#
# Solo descarga si el modelo paso el quality gate de finalize.sh.

set -euo pipefail

REMOTE="${1:?Usage: $0 user@host}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/lambda.pem}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Verifica el resultado del quality gate
echo "Verificando resultado en remoto..."
RESULT=$(ssh -i "$SSH_KEY" "$REMOTE" "cat /tmp/RESULT.txt 2>/dev/null || echo 'Status: NO_RESULT'")
echo "$RESULT"
echo ""

if echo "$RESULT" | grep -q "Status: *ACEPTADO"; then
    echo "Modelo aprobado. Descargando..."
else
    echo "ATENCION: el modelo NO paso el quality gate."
    read -p "Descargar de todos modos? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Cancelado."
        exit 1
    fi
fi

mkdir -p "$PROJECT_DIR/domains/chess/checkpoints" "$PROJECT_DIR/domains/chess/data"

scp -i "$SSH_KEY" \
    "$REMOTE:~/custom_llm/domains/chess/checkpoints/best_model.pt" \
    "$PROJECT_DIR/domains/chess/checkpoints/best_model.pt"

# Tambien el segundo lugar (phase1 o phase3) si existen, para tournament local
scp -i "$SSH_KEY" \
    "$REMOTE:~/custom_llm/domains/chess/checkpoints/phase1_model.pt" \
    "$PROJECT_DIR/domains/chess/checkpoints/phase1_model.pt" || true
scp -i "$SSH_KEY" \
    "$REMOTE:~/custom_llm/domains/chess/checkpoints/phase3_model.pt" \
    "$PROJECT_DIR/domains/chess/checkpoints/phase3_model.pt" || true

scp -i "$SSH_KEY" \
    "$REMOTE:~/custom_llm/domains/chess/data/vocab.json" \
    "$PROJECT_DIR/domains/chess/data/vocab.json"

# Logs para analisis post-mortem
scp -i "$SSH_KEY" "$REMOTE:/tmp/RESULT.txt" "$PROJECT_DIR/domains/chess/checkpoints/RESULT.txt" || true
scp -i "$SSH_KEY" "$REMOTE:/tmp/eval_phase1.log" "$PROJECT_DIR/domains/chess/checkpoints/eval_phase1.log" || true
scp -i "$SSH_KEY" "$REMOTE:/tmp/eval_phase3.log" "$PROJECT_DIR/domains/chess/checkpoints/eval_phase3.log" || true
scp -i "$SSH_KEY" "$REMOTE:/tmp/train.log" "$PROJECT_DIR/domains/chess/checkpoints/train_phase1.log" || true
scp -i "$SSH_KEY" "$REMOTE:/tmp/train_phase3.log" "$PROJECT_DIR/domains/chess/checkpoints/train_phase3.log" || true
scp -i "$SSH_KEY" "$REMOTE:/tmp/distill.log" "$PROJECT_DIR/domains/chess/checkpoints/distill.log" || true

echo ""
echo "Descarga completa:"
ls -lh "$PROJECT_DIR/domains/chess/checkpoints/best_model.pt" \
       "$PROJECT_DIR/domains/chess/data/vocab.json"
echo ""
echo "ACUERDATE de TERMINAR la instancia en lambdalabs.com"
