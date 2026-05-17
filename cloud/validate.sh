#!/usr/bin/env bash
# Validacion rapida en H100: 2 epochs sobre subset 30k, ~2 min.
# Si val_loss baja sin NaN -> lanzar train.sh con confianza.

set -euo pipefail

cd ~/custom_llm
source venv/bin/activate

python -c "
import torch, os
src = 'domains/chess/data/sequences.pt'
dst = 'domains/chess/data/sequences_validate.pt'
if not os.path.exists(dst):
    seqs = torch.load(src, weights_only=False)
    torch.save(seqs[:30000], dst)
    print(f'Subset creado: 30000 secuencias')
"

mv domains/chess/data/sequences.pt domains/chess/data/sequences_full.pt
cp domains/chess/data/sequences_validate.pt domains/chess/data/sequences.pt

# Mismos hiperparams que train.sh, solo 2 epochs
python cli.py chess train \
    --epochs 2 --patience 99 \
    --batch-size 384 \
    --d-model 384 --n-heads 8 --n-layers 8 \
    --max-len 192 \
    --lr 4.5e-4 \
    --warmup-steps 200

# Restaurar
mv domains/chess/data/sequences_full.pt domains/chess/data/sequences.pt
rm -f domains/chess/checkpoints/best_model.pt
echo ""
echo "Validacion completa. Si val_loss bajo de ~5 a <4 sin NaN -> lanzar train.sh"
