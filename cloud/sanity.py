"""
Sanity check cualitativo: juega partidas con el modelo entrenado y las imprime
para inspeccion visual. Complementa la evaluacion ELO (cuantitativa).

Uso: python cloud/sanity.py [--checkpoint PATH]

Comprueba:
  - El modelo juega partidas coherentes (no se cuelga, no repite trivialmente)
  - Las aperturas tienen sentido (lineas de ajedrez reales)
  - Largo de partida razonable (no flukes de 5 jugadas)
  - Mezcla de resultados (no siempre tablas por repeticion)
"""

import os
import sys
import argparse
import random
import chess
import chess.engine
import torch

DOMAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "domains", "chess")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.generator import load_model
from domains.chess.play import predict_chess_move, get_model_path, INFERENCE_ELO_BUCKET
from domains.chess.evaluator import find_stockfish
from core.dataset import BOS_TOKEN

DATA_DIR = os.path.join(DOMAIN_DIR, "data")


def play_selfplay(model, t2i, i2t, device, temperature, max_moves=200):
    """Modelo vs modelo. Devuelve (san_moves, result)."""
    board = chess.Board()
    token_ids = [t2i[BOS_TOKEN]]
    if INFERENCE_ELO_BUCKET in t2i:
        token_ids.append(t2i[INFERENCE_ELO_BUCKET])
    san_moves = []
    while not board.is_game_over() and len(san_moves) < max_moves:
        move = predict_chess_move(model, token_ids, t2i, i2t, board, device, temperature)
        san = board.san(move)
        san_moves.append(san)
        if san in t2i:
            token_ids.append(t2i[san])
        board.push(move)
    result = board.result() if board.is_game_over() else "*"
    return san_moves, result


def play_vs_stockfish(model, t2i, i2t, device, engine, skill, temperature, model_white, max_moves=200):
    """Modelo vs Stockfish. Devuelve (san_moves, result, model_score)."""
    engine.configure({"Skill Level": skill})
    board = chess.Board()
    token_ids = [t2i[BOS_TOKEN]]
    if INFERENCE_ELO_BUCKET in t2i:
        token_ids.append(t2i[INFERENCE_ELO_BUCKET])
    san_moves = []
    while not board.is_game_over() and len(san_moves) < max_moves:
        is_model_turn = (board.turn == chess.WHITE) == model_white
        if is_model_turn:
            move = predict_chess_move(model, token_ids, t2i, i2t, board, device, temperature)
        else:
            move = engine.play(board, chess.engine.Limit(depth=6)).move
        san = board.san(move)
        san_moves.append(san)
        if san in t2i:
            token_ids.append(t2i[san])
        board.push(move)
    result = board.result() if board.is_game_over() else "*"
    if board.is_game_over() and board.outcome() and board.outcome().winner is not None:
        won = board.outcome().winner == (chess.WHITE if model_white else chess.BLACK)
        score = 1.0 if won else 0.0
    else:
        score = 0.5
    return san_moves, result, score


def fmt_game(san_moves):
    """Formatea jugadas como '1. e4 e5 2. Nf3 ...'."""
    out = []
    for i, san in enumerate(san_moves):
        if i % 2 == 0:
            out.append(f"{i//2 + 1}.{san}")
        else:
            out.append(san)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Path al checkpoint (default: best_model.pt)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = args.checkpoint or get_model_path()
    model, t2i, i2t = load_model(vocab_path, model_path, device)

    print("=" * 60)
    print(f"  SANITY CHECK - {os.path.basename(model_path)}")
    print(f"  Device: {device}  Temp: {args.temperature}")
    print("=" * 60)

    all_lengths = []
    results = []

    # 1. Selfplay (modelo vs modelo)
    print("\n--- SELFPLAY (modelo vs modelo) ---")
    for g in range(4):
        san_moves, result = play_selfplay(model, t2i, i2t, device, args.temperature)
        all_lengths.append(len(san_moves))
        results.append(result)
        print(f"\nPartida {g+1}: {result}  ({len(san_moves)} jugadas)")
        print(f"  {fmt_game(san_moves[:40])}{'...' if len(san_moves) > 40 else ''}")

    # 2. Vs Stockfish
    sf = find_stockfish()
    if sf:
        print("\n--- VS STOCKFISH (skill 3, ~1200 ELO) ---")
        engine = chess.engine.SimpleEngine.popen_uci(sf)
        sf_scores = []
        for g in range(3):
            model_white = (g % 2 == 0)
            san_moves, result, score = play_vs_stockfish(
                model, t2i, i2t, device, engine, 3, args.temperature, model_white)
            all_lengths.append(len(san_moves))
            sf_scores.append(score)
            color = "blancas" if model_white else "negras"
            print(f"\nPartida {g+1} (modelo={color}): {result}  score modelo={score}  ({len(san_moves)} jugadas)")
            print(f"  {fmt_game(san_moves[:40])}{'...' if len(san_moves) > 40 else ''}")
        engine.quit()
        print(f"\n  Score vs Stockfish skill 3: {sum(sf_scores)}/{len(sf_scores)}")
    else:
        print("\n(Stockfish no encontrado, saltando partidas vs SF)")

    # 3. Veredicto de sanity
    print("\n" + "=" * 60)
    print("  VEREDICTO")
    print("=" * 60)
    avg_len = sum(all_lengths) / len(all_lengths)
    short_games = sum(1 for l in all_lengths if l < 15)
    print(f"  Largo promedio:        {avg_len:.0f} jugadas")
    print(f"  Partidas muy cortas:   {short_games}/{len(all_lengths)} (<15 jugadas)")
    print(f"  Resultados selfplay:   {results}")

    flags = []
    if avg_len < 20:
        flags.append("Partidas demasiado cortas - el modelo puede estar jugando mal")
    if short_games > len(all_lengths) // 2:
        flags.append("Muchas partidas cortas - revisar calidad")
    if len(set(results)) == 1 and results[0] == "1/2-1/2":
        flags.append("Todas tablas - posible repeticion trivial")

    if flags:
        print("\n  ATENCION:")
        for f in flags:
            print(f"    - {f}")
        print("\n  Las partidas NO se ven del todo sanas. Inspeccionar antes de aceptar.")
    else:
        print("\n  Las partidas se ven coherentes (largo razonable, variedad de resultados).")
        print("  Combinar con el ELO del quality gate para la decision final.")


if __name__ == "__main__":
    main()
