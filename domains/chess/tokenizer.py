"""
Tokenizador de ajedrez: convierte PGN en secuencias de tokens.
"""

import chess.pgn
from tqdm import tqdm

from core.dataset import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

# Tokens de resultado
RESULT_TOKENS = ["<WHITE_WINS>", "<BLACK_WINS>", "<DRAW>"]

RESULT_MAP = {
    "1-0": "<WHITE_WINS>",
    "0-1": "<BLACK_WINS>",
    "1/2-1/2": "<DRAW>",
}


def build_vocab(pgn_path, max_games=None):
    """Lee un archivo PGN y construye el vocabulario de movimientos."""
    moves = set()
    count = 0

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                san = board.san(move)
                moves.add(san)
                board.push(move)

            count += 1
            if max_games and count >= max_games:
                break

    special = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + RESULT_TOKENS
    vocab = special + sorted(moves)

    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}

    return token_to_id, id_to_token


def parse_games(pgn_path, token_to_id, max_games=None, min_moves=10):
    """Parsea partidas PGN y las convierte en secuencias de IDs de tokens."""
    sequences = []
    skipped = 0
    count = 0

    with open(pgn_path) as f:
        pbar = tqdm(desc="Parseando partidas")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result not in RESULT_MAP:
                skipped += 1
                pbar.update(1)
                continue

            board = game.board()
            move_tokens = []
            valid = True

            for move in game.mainline_moves():
                san = board.san(move)
                if san not in token_to_id:
                    valid = False
                    break
                move_tokens.append(token_to_id[san])
                board.push(move)

            if not valid or len(move_tokens) < min_moves:
                skipped += 1
                pbar.update(1)
                continue

            seq = (
                [token_to_id[BOS_TOKEN]]
                + move_tokens
                + [token_to_id[RESULT_MAP[result]]]
                + [token_to_id[EOS_TOKEN]]
            )
            sequences.append(seq)

            count += 1
            pbar.update(1)
            if max_games and count >= max_games:
                break

        pbar.close()

    print(f"Partidas parseadas: {count}, omitidas: {skipped}")
    return sequences
