"""
Jugar ajedrez contra el LLM o LLM vs LLM.
"""

import os
import chess
import random

from core.dataset import BOS_TOKEN
from core.generator import load_model, predict_next_token
from domains.chess.ui import render_board, get_human_move

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DOMAIN_DIR, "checkpoints")


def get_model_path(selftrained=False, rlhf=False):
    """Determina que checkpoint cargar."""
    if rlhf:
        filename = "best_model_rlhf.pt"
    elif selftrained:
        filename = "best_model_selftrained.pt"
    else:
        filename = "best_model.pt"

    path = os.path.join(CHECKPOINTS_DIR, filename)
    if not os.path.exists(path):
        print(f"{filename} no encontrado, usando modelo base.")
        path = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    return path


def predict_chess_move(model, token_ids, token_to_id, id_to_token, board, device, temperature=0.8):
    """Genera el siguiente movimiento de ajedrez, evitando repeticiones."""
    legal_moves = list(board.legal_moves)
    legal_sans = [board.san(m) for m in legal_moves]

    legal_ids = set()
    san_to_id = {}
    for san in legal_sans:
        if san in token_to_id:
            tid = token_to_id[san]
            legal_ids.add(tid)
            san_to_id[san] = tid

    if not legal_ids:
        return random.choice(legal_moves)

    # Check if a move would cause threefold repetition and exclude it
    non_repeating_ids = set()
    for san, tid in san_to_id.items():
        move = board.parse_san(san)
        board.push(move)
        if not board.can_claim_threefold_repetition():
            non_repeating_ids.add(tid)
        board.pop()

    # Use non-repeating moves if available, otherwise allow all
    valid_ids = non_repeating_ids if non_repeating_ids else legal_ids

    token_id = predict_next_token(model, token_ids, valid_ids, device, temperature)
    san = id_to_token[token_id]
    return board.parse_san(san)


def play(color="white", temperature=0.8, selftrained=False, rlhf=False):
    """Juego interactivo humano vs LLM."""
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = get_model_path(selftrained=selftrained, rlhf=rlhf)
    model, token_to_id, id_to_token = load_model(vocab_path, model_path, device)

    board = chess.Board()
    human_white = (color == "white")
    token_ids = [token_to_id[BOS_TOKEN]]

    print(f"\n{'='*40}")
    print(f"  AJEDREZ - Tu juegas con {'BLANCAS' if human_white else 'NEGRAS'}")
    print(f"  Escribe movimientos en notacion algebraica")
    print(f"  Ejemplos: e4, Nf3, O-O, Bxe5, e8=Q")
    print(f"  Escribe 'salir' para terminar")
    print(f"{'='*40}")

    while not board.is_game_over():
        render_board(board, perspective_white=human_white)

        is_white_turn = board.turn == chess.WHITE
        is_human_turn = (human_white and is_white_turn) or (not human_white and not is_white_turn)
        turn_label = "Blancas" if is_white_turn else "Negras"

        if is_human_turn:
            move = get_human_move(board, turn_label)
            if move is None:
                print("Partida abandonada.")
                return
            san = board.san(move)
        else:
            move = predict_chess_move(model, token_ids, token_to_id, id_to_token, board, device, temperature)
            san = board.san(move)
            print(f"  [{turn_label}] LLM juega: {san}")

        if san in token_to_id:
            token_ids.append(token_to_id[san])
        board.push(move)

    render_board(board, perspective_white=human_white)
    result = board.result()
    print(f"\nResultado: {result}")

    if result == "1-0":
        print("Blancas ganan!" if human_white else "El LLM gana!")
    elif result == "0-1":
        print("Negras ganan!" if not human_white else "El LLM gana!")
    else:
        print("Empate!")
