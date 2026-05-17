#!/usr/bin/env bash
# Setup completo en instancia Lambda (A100 40GB SXM4). Hace todo: deps + datos + prepare.
# Asume Ubuntu 22.04 con NVIDIA drivers + CUDA 12.x ya instalados (default Lambda).
# Idempotente: se puede re-ejecutar.

set -euo pipefail

cd ~/custom_llm

# 1. venv aislado
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip --quiet
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 --quiet
pip install -r cloud/requirements.txt --no-deps --quiet

# 2. Verificacion CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NO disponible'; \
print(f'CUDA OK: {torch.cuda.get_device_name(0)}, compute {torch.cuda.get_device_capability(0)}, memoria {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')"

# 2b. Stockfish para evaluacion ELO post-training
if ! command -v stockfish &> /dev/null; then
    echo "Instalando Stockfish..."
    sudo apt-get update -qq && sudo apt-get install -y stockfish -qq
fi
# NO hacer 'stockfish --version | head': stockfish entra en modo UCI y el SIGPIPE
# de head mata el script con pipefail. Solo confirmamos que el binario existe.
echo "Stockfish instalado: $(command -v stockfish)"

# 3. Descarga del dump Lichess si no existe
LICHESS_URL="https://database.lichess.org/standard/lichess_db_standard_rated_2022-11.pgn.zst"
ZST_FILE="/tmp/lichess_db_standard_rated_2022-11.pgn.zst"

if [ ! -f "$ZST_FILE" ]; then
    echo ""
    echo "Descargando dump Lichess (27 GB, ~5-10 min en cloud)..."
    curl -L --progress-bar -o "$ZST_FILE" "$LICHESS_URL"
fi
echo "Dump: $(du -h $ZST_FILE | awk '{print $1}')"

# 4. Prepare dataset 1M partidas si no existe
SEQ_FILE="domains/chess/data/sequences.pt"
if [ ! -f "$SEQ_FILE" ] || [ "$(python -c 'import torch; print(len(torch.load("'$SEQ_FILE'", weights_only=False)))')" -lt 800000 ]; then
    echo ""
    echo "Preparando dataset (1M partidas, min ELO 1600)..."
    python cli.py chess prepare --max-games 1000000 --min-elo 1600
else
    echo "Dataset ya preparado: $(python -c 'import torch; print(len(torch.load("'$SEQ_FILE'", weights_only=False)))') secuencias"
fi

echo ""
echo "Setup completo. Listo para validar/entrenar:"
echo "  bash cloud/validate.sh    # 2 epochs sanity check (~3 min)"
echo "  bash cloud/train.sh       # training final (~3-5h)"
