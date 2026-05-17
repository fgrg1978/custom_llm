"""
Tokenizador de ajedrez: convierte PGN en secuencias de tokens.

Filtrado de calidad (Fase 1):
  - min_elo: descarta partidas donde algun jugador esta por debajo del umbral.
  - ELO bucket tokens: cada partida se prefija con <ELO_XXXX> segun la fuerza
    del jugador mas debil. Permite condicionar generacion a nivel de juego.
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

# Buckets de ELO (del mas bajo al mas alto). Cada partida se etiqueta con el
# bucket correspondiente al min(WhiteElo, BlackElo).
ELO_BUCKETS = [
    (1800, "<ELO_1800>"),
    (2000, "<ELO_2000>"),
    (2200, "<ELO_2200>"),
    (2400, "<ELO_2400>"),
]
ELO_BUCKET_TOKENS = [t for _, t in ELO_BUCKETS]


def elo_bucket_token(white_elo, black_elo):
    """Token de bucket para la partida, basado en el jugador mas debil."""
    lower = min(white_elo, black_elo)
    chosen = ELO_BUCKETS[0][1]
    for threshold, token in ELO_BUCKETS:
        if lower >= threshold:
            chosen = token
    return chosen


def _read_elos(game):
    """Extrae (WhiteElo, BlackElo) del PGN. Devuelve (None, None) si faltan."""
    try:
        return int(game.headers.get("WhiteElo", "?")), int(game.headers.get("BlackElo", "?"))
    except ValueError:
        return None, None


def _open(pgn_source):
    """Devuelve (file_obj, should_close). pgn_source puede ser path o stream."""
    if isinstance(pgn_source, str):
        return open(pgn_source), True
    return pgn_source, False


def build_vocab(pgn_source, max_games=None, min_elo=2000):
    """Construye vocabulario a partir de partidas que pasan el filtro de ELO.

    pgn_source: ruta (str) o file-like object ya abierto.
    """
    moves = set()
    count = 0

    f, should_close = _open(pgn_source)
    try:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            white_elo, black_elo = _read_elos(game)
            if white_elo is None or min(white_elo, black_elo) < min_elo:
                continue

            board = game.board()
            for move in game.mainline_moves():
                san = board.san(move)
                moves.add(san)
                board.push(move)

            count += 1
            if max_games and count >= max_games:
                break
    finally:
        if should_close:
            f.close()

    special = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + RESULT_TOKENS + ELO_BUCKET_TOKENS
    vocab = special + sorted(moves)

    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}

    return token_to_id, id_to_token


def parse_games(pgn_source, token_to_id, max_games=None, min_moves=10, min_elo=2000):
    """Parsea PGN a secuencias de IDs, filtrando por ELO y resultado.

    pgn_source: ruta (str) o file-like object ya abierto.
    """
    sequences = []
    skipped_result = 0
    skipped_elo = 0
    skipped_moves = 0
    count = 0

    f, should_close = _open(pgn_source)
    try:
        pbar = tqdm(desc="Parseando partidas")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            pbar.update(1)

            white_elo, black_elo = _read_elos(game)
            if white_elo is None or min(white_elo, black_elo) < min_elo:
                skipped_elo += 1
                continue

            result = game.headers.get("Result", "*")
            if result not in RESULT_MAP:
                skipped_result += 1
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
                skipped_moves += 1
                continue

            elo_tok = elo_bucket_token(white_elo, black_elo)
            seq = (
                [token_to_id[BOS_TOKEN]]
                + [token_to_id[elo_tok]]
                + move_tokens
                + [token_to_id[RESULT_MAP[result]]]
                + [token_to_id[EOS_TOKEN]]
            )
            sequences.append(seq)

            count += 1
            if max_games and count >= max_games:
                break

        pbar.close()
    finally:
        if should_close:
            f.close()

    print(f"Partidas parseadas: {count}")
    print(f"  Omitidas por ELO < {min_elo}: {skipped_elo}")
    print(f"  Omitidas por resultado invalido: {skipped_result}")
    print(f"  Omitidas por pocas jugadas / SAN desconocido: {skipped_moves}")
    return sequences


def build_vocab_and_parse(pgn_source, max_games=None, min_moves=10, min_elo=2000):
    """Pasada UNICA: lee el PGN una vez, construye vocab y tokeniza.

    Sustituye a build_vocab + parse_games (que escaneaban el .zst dos veces).
    El parsing de PGN + descompresion zstd es el cuello de botella; hacerlo una
    sola vez ~duplica la velocidad de preparacion.

    Devuelve (token_to_id, id_to_token, sequences).
    """
    raw_games = []  # (elo_tok, [san_moves], result) de cada partida valida
    moves_set = set()
    skipped_elo = skipped_result = skipped_moves = 0
    count = 0

    f, should_close = _open(pgn_source)
    try:
        pbar = tqdm(desc="Parseando partidas (pasada unica)")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            pbar.update(1)

            white_elo, black_elo = _read_elos(game)
            if white_elo is None or min(white_elo, black_elo) < min_elo:
                skipped_elo += 1
                continue

            result = game.headers.get("Result", "*")
            if result not in RESULT_MAP:
                skipped_result += 1
                continue

            board = game.board()
            san_moves = []
            for move in game.mainline_moves():
                san_moves.append(board.san(move))
                board.push(move)

            if len(san_moves) < min_moves:
                skipped_moves += 1
                continue

            moves_set.update(san_moves)
            elo_tok = elo_bucket_token(white_elo, black_elo)
            raw_games.append((elo_tok, san_moves, result))

            count += 1
            if max_games and count >= max_games:
                break

        pbar.close()
    finally:
        if should_close:
            f.close()

    # Vocab a partir de lo recolectado
    special = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + RESULT_TOKENS + ELO_BUCKET_TOKENS
    vocab = special + sorted(moves_set)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}

    # Tokenizacion en memoria (rapido: solo lookups de dict, sin re-parsear)
    bos, eos = token_to_id[BOS_TOKEN], token_to_id[EOS_TOKEN]
    sequences = []
    for elo_tok, san_moves, result in raw_games:
        seq = (
            [bos, token_to_id[elo_tok]]
            + [token_to_id[s] for s in san_moves]
            + [token_to_id[RESULT_MAP[result]], eos]
        )
        sequences.append(seq)

    print(f"Partidas parseadas: {count}")
    print(f"  Omitidas por ELO < {min_elo}: {skipped_elo}")
    print(f"  Omitidas por resultado invalido: {skipped_result}")
    print(f"  Omitidas por pocas jugadas: {skipped_moves}")
    print(f"Vocabulario: {len(token_to_id)} tokens")
    return token_to_id, id_to_token, sequences
