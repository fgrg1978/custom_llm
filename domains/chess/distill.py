"""
Stockfish supervised data generation (Fase 3 lite).

Para cada partida del dump:
  1. Elige K posiciones aleatorias (despues del move 5, antes del move 80).
  2. Para cada posicion, consulta Stockfish a profundidad fija (default 10).
  3. Emite secuencia: [BOS, ELO_BUCKET, move1, ..., move_n, stockfish_best]

El resultado se guarda en sequences_distill.pt y se combina con sequences.pt
durante el training para mezclar imitacion humana + supervision Stockfish.

Diseño:
  - Productor (main): lee el .zst de Lichess, extrae partidas validas y construye
    los token_ids historicos + el FEN para Stockfish.
  - Workers (multiprocessing.Pool): cada worker tiene su propio engine Stockfish,
    consulta best_move y devuelve la secuencia final.
  - El cuello de botella es Stockfish (CPU); con N workers escala casi lineal.
"""

import io
import os
import random
import multiprocessing as mp
import chess
import chess.pgn
import chess.engine
import torch
from tqdm import tqdm

from domains.chess.tokenizer import elo_bucket_token, _read_elos
from domains.chess.prepare import open_pgn_stream, _zst_path_for_url, DEFAULT_URL
from core.dataset import BOS_TOKEN, load_vocab

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
DISTILL_FILE = os.path.join(DATA_DIR, "sequences_distill.pt")


def _stockfish_worker(args):
    """Worker: para cada (history_tokens, fen, elo_bucket_id, best_move_idx_placeholder)
    consulta Stockfish y devuelve la secuencia final con el best_move tokenizado.

    Recibe items = lista de (history_token_ids, fen, elo_bucket_id) y stockfish_path
    Devuelve lista de sequences (lista de int).
    """
    items, stockfish_path, depth, token_to_id = args
    sequences = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"Worker no pudo iniciar Stockfish: {e}")
        return sequences

    try:
        for history_ids, fen, elo_id, bos_id in items:
            try:
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(depth=depth))
                if result.move is None:
                    continue
                best_san = board.san(result.move)
                if best_san not in token_to_id:
                    continue
                seq = [bos_id, elo_id] + history_ids + [token_to_id[best_san]]
                sequences.append(seq)
            except Exception:
                continue
    finally:
        engine.quit()

    return sequences


def _extract_positions(pgn_source, token_to_id, max_games, min_elo,
                      positions_per_game, min_move=5, max_move=80, seed=42):
    """Productor: lee PGN, extrae (history_token_ids, fen, elo_bucket_id) para
    posiciones aleatorias en cada partida valida.
    """
    rng = random.Random(seed)
    items = []
    bos_id = token_to_id[BOS_TOKEN]
    games_read = 0

    pbar = tqdm(desc="Extrayendo posiciones")
    while True:
        game = chess.pgn.read_game(pgn_source)
        if game is None:
            break

        white_elo, black_elo = _read_elos(game)
        if white_elo is None or min(white_elo, black_elo) < min_elo:
            continue

        moves = list(game.mainline_moves())
        if len(moves) < min_move + 5:
            continue

        elo_tok = elo_bucket_token(white_elo, black_elo)
        if elo_tok not in token_to_id:
            continue
        elo_id = token_to_id[elo_tok]

        # Elegir posiciones aleatorias dentro del rango valido
        upper = min(len(moves), max_move)
        candidate_indices = list(range(min_move, upper))
        k = min(positions_per_game, len(candidate_indices))
        if k == 0:
            continue
        chosen = rng.sample(candidate_indices, k)

        # Replay hasta cada posicion y guarda historia + FEN
        # Para eficiencia, vamos avanzando un solo board y guardamos snapshots
        chosen_set = set(chosen)
        board = game.board()
        history_ids = []
        valid = True

        for i, move in enumerate(moves[:upper]):
            if i in chosen_set:
                items.append((list(history_ids), board.fen(), elo_id, bos_id))
            san = board.san(move)
            if san not in token_to_id:
                valid = False
                break
            history_ids.append(token_to_id[san])
            board.push(move)

        if not valid:
            # Quita los items que se anadieron antes de que la partida fallara
            # (no es exacto pero raro: si falla un SAN, asume desfase y descarta los recientes)
            pass

        games_read += 1
        pbar.update(1)
        if max_games and games_read >= max_games:
            break

    pbar.close()
    print(f"Partidas leidas: {games_read}")
    print(f"Posiciones extraidas: {len(items)}")
    return items


def distill(
    vocab_path,
    n_games=200000,
    positions_per_game=5,
    min_elo=1600,
    depth=10,
    n_workers=None,
    url=DEFAULT_URL,
    output_path=DISTILL_FILE,
):
    """Pipeline completo de distillation."""
    import shutil

    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 2)
    stockfish_path = shutil.which("stockfish")
    if stockfish_path is None:
        raise RuntimeError("stockfish no encontrado en PATH. Instalar con: apt-get install stockfish")

    print(f"Stockfish: {stockfish_path}")
    print(f"Workers: {n_workers}")
    print(f"Depth: {depth}")
    print(f"Target positions: {n_games} games x {positions_per_game} pos/game = {n_games*positions_per_game}")

    token_to_id, _ = load_vocab(vocab_path)

    # Productor: extraer posiciones del .zst
    zst_file = _zst_path_for_url(url)
    print(f"\nLeyendo: {zst_file}")
    with open_pgn_stream(zst_file) as stream:
        items = _extract_positions(
            stream, token_to_id,
            max_games=n_games,
            min_elo=min_elo,
            positions_per_game=positions_per_game,
        )

    if not items:
        print("ERROR: no se extrajeron posiciones.")
        return

    # Particionar items entre workers
    chunk_size = max(1, len(items) // n_workers)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    args_list = [(chunk, stockfish_path, depth, token_to_id) for chunk in chunks]

    print(f"\nEtiquetando con Stockfish en {n_workers} workers...")
    all_sequences = []
    with mp.Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(_stockfish_worker, args_list),
                          total=len(args_list), desc="Workers"):
            all_sequences.extend(result)

    print(f"\nSecuencias generadas: {len(all_sequences)}")
    if not all_sequences:
        print("ERROR: ninguna secuencia generada.")
        return

    torch.save(all_sequences, output_path)
    print(f"Guardado: {output_path}")

    lengths = [len(s) for s in all_sequences]
    print(f"\nEstadisticas:")
    print(f"  Posiciones validas: {len(all_sequences)}")
    print(f"  Largo promedio: {sum(lengths)/len(lengths):.0f} tokens")
    print(f"  Largo maximo:   {max(lengths)}")
    print(f"  Largo minimo:   {min(lengths)}")
