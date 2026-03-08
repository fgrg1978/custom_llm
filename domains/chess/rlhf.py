"""
RLHF para ajedrez: feedback manual o automatico (Stockfish).
"""

import os
import chess
import torch
from tqdm import tqdm

from core.dataset import BOS_TOKEN
from core.generator import load_model
from core.rlhf import rlhf_train
from domains.chess.play import predict_chess_move, get_model_path
from domains.chess.evaluator import StockfishEvaluator, HeuristicEvaluator, find_stockfish
from domains.chess.ui import render_board, get_human_move

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DOMAIN_DIR, "checkpoints")


def play_with_feedback_manual(model, token_to_id, id_to_token, device, human_color="black", temperature=0.8):
    """Partida interactiva con feedback humano."""
    board = chess.Board()
    human_white = (human_color == "white")
    token_ids = [token_to_id[BOS_TOKEN]]
    experiences = []

    print(f"\n{'='*50}")
    print(f"  RLHF - Feedback Manual")
    print(f"  Tu juegas con {'BLANCAS' if human_white else 'NEGRAS'}")
    print(f"  Despues de cada movimiento del LLM, califica:")
    print(f"    [b] bueno  [m] malo  [enter] neutral")
    print(f"{'='*50}")

    while not board.is_game_over():
        render_board(board, perspective_white=human_white)

        is_white_turn = board.turn == chess.WHITE
        is_human_turn = (human_white and is_white_turn) or (not human_white and not is_white_turn)
        turn_label = "Blancas" if is_white_turn else "Negras"

        if is_human_turn:
            move = get_human_move(board, turn_label)
            if move is None:
                return experiences
            san = board.san(move)
        else:
            move = predict_chess_move(model, token_ids, token_to_id, id_to_token, board, device, temperature)
            san = board.san(move)
            print(f"  [{turn_label}] LLM juega: {san}")

            fb = input(f"  Califica [b]ueno / [m]alo / [enter] neutral: ").strip().lower()
            if fb == "b":
                reward = 1.0
                print(f"  -> Reward: +1.0")
            elif fb == "m":
                reward = -1.0
                print(f"  -> Reward: -1.0")
            else:
                reward = 0.0

            if san in token_to_id:
                experiences.append((list(token_ids), token_to_id[san], reward))

        if san in token_to_id:
            token_ids.append(token_to_id[san])
        board.push(move)

    result = board.result()
    render_board(board, perspective_white=human_white)
    print(f"\nResultado: {result}")

    llm_is_white = not human_white
    result_reward = 0
    if result == "1-0":
        result_reward = 5.0 if llm_is_white else -5.0
    elif result == "0-1":
        result_reward = 5.0 if not llm_is_white else -5.0

    if result_reward != 0:
        for i in range(len(experiences)):
            tid, tar, rew = experiences[i]
            experiences[i] = (tid, tar, rew + result_reward)
        print(f"  Bonus resultado ({result_reward:+.1f}) aplicado a {len(experiences)} movimientos")

    return experiences


def play_with_feedback_auto(model, token_to_id, id_to_token, device, evaluator, temperature=0.8):
    """Partida automatica donde evaluador da el feedback."""
    board = chess.Board()
    token_ids = [token_to_id[BOS_TOKEN]]
    experiences = []

    while not board.is_game_over() and len(experiences) < 200:
        board_before = board.copy()

        move = predict_chess_move(model, token_ids, token_to_id, id_to_token, board, device, temperature)
        san = board.san(move)

        if san in token_to_id:
            board.push(move)
            reward = evaluator.get_reward(board_before, board)
            experiences.append((list(token_ids), token_to_id[san], reward))
            token_ids.append(token_to_id[san])
        else:
            board.push(move)
            token_ids.append(0)

    result = board.result() if board.is_game_over() else "*"

    for i in range(len(experiences)):
        tid, tar, rew = experiences[i]
        if result == "1-0":
            bonus = 3.0 if i % 2 == 0 else -3.0
        elif result == "0-1":
            bonus = -3.0 if i % 2 == 0 else 3.0
        else:
            bonus = 0
        experiences[i] = (tid, tar, rew + bonus)

    return experiences, result


def run_rlhf(feedback="auto", n_games=50, rounds=5, lr=5e-5, temperature=0.8, selftrained=False):
    """Loop principal de RLHF."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = get_model_path(selftrained=selftrained)
    model, token_to_id, id_to_token = load_model(vocab_path, model_path, device)
    vocab_size = len(token_to_id)

    evaluator = None
    if feedback == "auto":
        if find_stockfish():
            evaluator = StockfishEvaluator(depth=10, time_limit=0.05)
            print(f"Usando Stockfish")
        else:
            print("Stockfish no encontrado, usando heuristico.")
            evaluator = HeuristicEvaluator()
    elif feedback == "heuristic":
        evaluator = HeuristicEvaluator()

    print(f"\n{'='*50}")
    print(f"  RLHF - {'Manual' if feedback == 'manual' else 'Automatico'}")
    print(f"  Rondas: {rounds}, Partidas/ronda: {n_games}")
    print(f"  Dispositivo: {device}")
    print(f"{'='*50}\n")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    for round_num in range(1, rounds + 1):
        print(f"\n{'─'*50}")
        print(f"RONDA {round_num}/{rounds}")
        print(f"{'─'*50}")

        all_experiences = []

        if feedback == "manual":
            exps = play_with_feedback_manual(model, token_to_id, id_to_token, device, temperature=temperature)
            all_experiences.extend(exps)
        else:
            results = {}
            for _ in tqdm(range(n_games), desc="Jugando"):
                exps, result = play_with_feedback_auto(
                    model, token_to_id, id_to_token, device, evaluator, temperature=temperature
                )
                all_experiences.extend(exps)
                results[result] = results.get(result, 0) + 1
            print(f"  Resultados: {results}")

        if all_experiences:
            rewards = [r for _, _, r in all_experiences]
            avg_reward = sum(rewards) / len(rewards)
            pos = sum(1 for r in rewards if r > 0)
            neg = sum(1 for r in rewards if r < 0)
            print(f"  Experiencias: {len(all_experiences)}")
            print(f"  Reward promedio: {avg_reward:.3f}")
            print(f"  Positivos: {pos}, Negativos: {neg}")

        print(f"\n  Entrenando...")
        model = rlhf_train(model, all_experiences, vocab_size, device, lr=lr)

        checkpoint = {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "d_model": 128, "n_heads": 4, "n_layers": 4, "max_len": 256,
            "round": round_num, "training": f"rlhf_{feedback}",
            "epoch": round_num, "val_loss": 0,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, "best_model_rlhf.pt"))
        print(f"  Modelo guardado")

    if evaluator:
        evaluator.close()

    print(f"\n{'='*50}")
    print(f"  RLHF COMPLETADO")
    print(f"{'='*50}")
