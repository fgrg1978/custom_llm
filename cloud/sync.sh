#!/usr/bin/env bash
# Sincroniza solo el codigo a la instancia Lambda (~3 MB).
# Los datos (.zst, sequences.pt) se descargan/preparan en el H100 via setup.sh.
# Uso: SSH_KEY=~/.ssh/lambda.pem ./cloud/sync.sh ubuntu@LAMBDA_IP

set -euo pipefail

REMOTE="${1:?Usage: $0 user@host}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/lambda.pem}"

# Ruta absoluta del proyecto (rsync rechaza '..' en pathnames)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    --exclude 'venv/' \
    --exclude 'cloud/venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'experiments/*/' \
    --exclude 'domains/chess/data/games.pgn' \
    --exclude 'domains/chess/data/games.pgn.zst' \
    --exclude 'domains/chess/data/sequences.pt' \
    --exclude 'domains/chess/data/sequences*.pt' \
    --exclude 'domains/chess/checkpoints/' \
    "$PROJECT_DIR/" "$REMOTE:~/custom_llm/"

echo "Sync completo a $REMOTE:~/custom_llm/"
