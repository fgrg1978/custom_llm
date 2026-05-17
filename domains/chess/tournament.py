"""
Torneo head-to-head entre dos checkpoints del LLM.
Util para validar cambios sin depender de Stockfish: si tras un retrain
el modelo nuevo gana al antiguo con significancia, la mejora es real.
"""

import os
import math
import random
import chess
import torch
from tqdm import tqdm

from core.dataset import BOS_TOKEN
from core.generator import load_model
from domains.chess.play import predict_chess_move

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DOMAIN_DIR, "checkpoints")

CHECKPOINT_NAMES = {
    "base": "best_model.pt",
    "selftrained": "best_model_selftrained.pt",
    "rlhf": "best_model_rlhf.pt",
}


def resolve_checkpoint(spec):
    """Resuelve 'base'/'selftrained'/'rlhf' o una ruta directa a path absoluto."""
    if spec in CHECKPOINT_NAMES:
        path = os.path.join(CHECKPOINTS_DIR, CHECKPOINT_NAMES[spec])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint '{spec}' no encontrado en {path}")
        return path
    if os.path.exists(spec):
        return spec
    raise FileNotFoundError(f"Checkpoint no encontrado: {spec}")


def play_match(model_a, model_b, token_to_id, id_to_token, device,
               a_is_white, temperature, max_moves=200):
    """Partida LLM_A vs LLM_B. Devuelve score desde perspectiva de A."""
    board = chess.Board()
    token_ids = [token_to_id[BOS_TOKEN]]
    moves = 0

    while not board.is_game_over() and moves < max_moves:
        is_white_turn = board.turn == chess.WHITE
        a_to_move = (is_white_turn == a_is_white)
        model = model_a if a_to_move else model_b

        move = predict_chess_move(
            model, token_ids, token_to_id, id_to_token,
            board, device, temperature,
        )
        san = board.san(move)
        if san in token_to_id:
            token_ids.append(token_to_id[san])
        board.push(move)
        moves += 1

    if not board.is_game_over():
        return 0.5
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.5
    if outcome.winner == chess.WHITE:
        return 1.0 if a_is_white else 0.0
    return 0.0 if a_is_white else 1.0


def elo_delta(score, n_games):
    """Delta de ELO relativa (A vs B) a partir del score. CI al 95%."""
    if score <= 0:
        return -400.0, -600.0, -200.0
    if score >= 1:
        return 400.0, 200.0, 600.0
    delta = 400 * math.log10(score / (1 - score))
    se = math.sqrt(score * (1 - score) / n_games)
    s_lo = max(0.01, score - 1.96 * se)
    s_hi = min(0.99, score + 1.96 * se)
    return (
        delta,
        400 * math.log10(s_lo / (1 - s_lo)),
        400 * math.log10(s_hi / (1 - s_hi)),
    )


def tournament(a="base", b="rlhf", n_games=50, temperature=0.3, seed=42):
    """Torneo head-to-head entre dos checkpoints. Colores alternados."""
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vocab_path = os.path.join(DATA_DIR, "vocab.json")

    path_a = resolve_checkpoint(a)
    path_b = resolve_checkpoint(b)

    print(f"\n{'='*60}")
    print(f"  Torneo head-to-head")
    print(f"  A: {os.path.basename(path_a)}")
    print(f"  B: {os.path.basename(path_b)}")
    print(f"  Partidas: {n_games} (alternando colores, seed={seed})")
    print(f"{'='*60}\n")

    model_a, token_to_id, id_to_token = load_model(vocab_path, path_a, device)
    model_b, _, _ = load_model(vocab_path, path_b, device)

    wins_a = draws = wins_b = 0
    score = 0.0

    pbar = tqdm(range(n_games), desc=f"{a} vs {b}")
    for i in pbar:
        a_is_white = (i % 2 == 0)
        r = play_match(model_a, model_b, token_to_id, id_to_token,
                       device, a_is_white, temperature)
        score += r
        if r == 1.0:
            wins_a += 1
        elif r == 0.5:
            draws += 1
        else:
            wins_b += 1
        pbar.set_postfix(A=wins_a, D=draws, B=wins_b)

    score_pct = score / n_games
    delta, lo, hi = elo_delta(score_pct, n_games)

    print(f"\n{'='*60}")
    print(f"  Resultado")
    print(f"{'='*60}")
    print(f"  A wins: {wins_a:>3}")
    print(f"  Draws:  {draws:>3}")
    print(f"  B wins: {wins_b:>3}")
    print(f"  Score A: {score_pct:.1%}")
    print(f"  Delta ELO (A - B): {delta:+.0f}  95%CI [{lo:+.0f}, {hi:+.0f}]")

    # Significancia: CI no cruza 0
    if lo > 0:
        verdict = f"A es mejor que B por ~{delta:.0f} ELO (significativo)"
    elif hi < 0:
        verdict = f"B es mejor que A por ~{-delta:.0f} ELO (significativo)"
    else:
        verdict = "Sin diferencia significativa (CI cruza 0)"
    print(f"\n  {verdict}")
    print(f"{'='*60}\n")

    return {
        "a": a, "b": b,
        "wins_a": wins_a, "draws": draws, "wins_b": wins_b,
        "score_a": score_pct,
        "delta_elo": delta, "ci_lo": lo, "ci_hi": hi,
        "verdict": verdict,
    }
