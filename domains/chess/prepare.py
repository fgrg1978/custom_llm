"""
Descarga y prepara datos de partidas de ajedrez desde Lichess.
"""

import os
import subprocess
import sys

from domains.chess.tokenizer import build_vocab, parse_games
from core.dataset import save_vocab

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
PGN_FILE = os.path.join(DATA_DIR, "games.pgn")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")
DATA_FILE = os.path.join(DATA_DIR, "sequences.pt")

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"


def download_pgn():
    """Descarga y descomprime un archivo PGN de Lichess."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PGN_FILE):
        print(f"Archivo PGN ya existe: {PGN_FILE}")
        return

    zst_file = PGN_FILE + ".zst"

    if not os.path.exists(zst_file):
        print(f"Descargando partidas de Lichess...")
        print(f"URL: {LICHESS_URL}")
        subprocess.run(["/usr/bin/curl", "-L", "-o", zst_file, LICHESS_URL], check=True)

    print("Descomprimiendo...")
    try:
        subprocess.run(["/opt/homebrew/bin/zstd", "-d", zst_file, "-o", PGN_FILE], check=True)
    except FileNotFoundError:
        print("zstd no encontrado. Instalalo con: brew install zstd")
        sys.exit(1)

    print(f"PGN listo: {PGN_FILE}")


def prepare(max_games=50000):
    """Pipeline completo de preparacion de datos."""
    import torch

    download_pgn()

    print(f"\nConstruyendo vocabulario (max {max_games} partidas)...")
    token_to_id, id_to_token = build_vocab(PGN_FILE, max_games=max_games)
    save_vocab(token_to_id, VOCAB_FILE)
    print(f"Vocabulario: {len(token_to_id)} tokens")

    print(f"\nParseando partidas...")
    sequences = parse_games(PGN_FILE, token_to_id, max_games=max_games)
    torch.save(sequences, DATA_FILE)
    print(f"Secuencias guardadas: {len(sequences)}")

    lengths = [len(s) for s in sequences]
    print(f"\nEstadisticas:")
    print(f"  Partidas: {len(sequences)}")
    print(f"  Largo promedio: {sum(lengths)/len(lengths):.0f} movimientos")
    print(f"  Largo maximo: {max(lengths)}")
    print(f"  Largo minimo: {min(lengths)}")
