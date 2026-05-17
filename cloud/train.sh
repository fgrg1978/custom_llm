#!/usr/bin/env bash
# Training final en A100 40GB SXM4. ~3-5h, ~$6-10 a $1.99/h.
# Plan B+ "production grade": 1M games + 25M params + 50 epochs.

set -euo pipefail

cd ~/custom_llm
source venv/bin/activate

# Sanity: confirma dataset esperado (>=800k partidas)
python -c "
import torch
seqs = torch.load('domains/chess/data/sequences.pt', weights_only=False)
n = len(seqs)
print(f'Dataset: {n:,} secuencias')
assert n >= 800000, f'Dataset solo tiene {n} secuencias. Re-ejecuta setup.sh primero.'
"

mkdir -p domains/chess/checkpoints
[ -f domains/chess/checkpoints/best_model.pt ] && \
    cp domains/chess/checkpoints/best_model.pt domains/chess/checkpoints/best_model.pt.bak || true

# A100 40GB SXM4 optimizado:
#  - 1M+ games (preparado por setup.sh)
#  - 25M params: d_model=384, n_heads=8, n_layers=8 (~10 GB con activaciones, sobra en 40GB)
#  - batch=384: comodo en 40GB para este modelo pequeno
#  - lr=4.5e-4: sqrt-scaling con margen de seguridad
#  - warmup=2000: ramp-up suave
#  - 50 epochs, patience=8 (corta antes si plateau)
#  - max_len=192: cubre >97% de partidas
#  - bf16 mixed precision (automatico en CUDA, A100 Ampere soporta bf16 nativo)

python cli.py chess train \
    --epochs 50 --patience 8 \
    --batch-size 384 \
    --d-model 384 --n-heads 8 --n-layers 8 \
    --max-len 192 \
    --lr 4.5e-4 \
    --warmup-steps 2000 \
    2>&1 | tee /tmp/train.log

echo ""
echo "Training completo."
echo "Checkpoint:    domains/chess/checkpoints/best_model.pt"
echo "Vocab:         domains/chess/data/vocab.json"
echo ""
echo "Descargar al local (desde tu Mac):"
echo "  scp -i \$SSH_KEY ubuntu@HOST:~/custom_llm/domains/chess/checkpoints/best_model.pt domains/chess/checkpoints/"
echo "  scp -i \$SSH_KEY ubuntu@HOST:~/custom_llm/domains/chess/data/vocab.json domains/chess/data/"
echo ""
echo "Luego TERMINA la instancia en lambdalabs.com"
