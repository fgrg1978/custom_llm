"""
RLHF for chess: manual or automatic feedback (Stockfish).

Inspired by Karpathy's nanochat REINFORCE:
- Multiple samples per position (try N moves, compare rewards)
- Advantage normalization (reward - mean)
- On-policy (fresh games each round)
"""

import os
import chess
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.dataset import BOS_TOKEN
from core.generator import load_model, predict_next_token
from core.rlhf import rlhf_train
from domains.chess.play import predict_chess_move, get_model_path, INFERENCE_ELO_BUCKET
from domains.chess.evaluator import StockfishEvaluator, HeuristicEvaluator, find_stockfish
from domains.chess.ui import render_board, get_human_move

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DOMAIN_DIR, "checkpoints")


def sample_multiple_moves(model, token_ids, token_to_id, id_to_token, board, evaluator, device, num_samples=8, temperature=1.0):
    """
    Generate N candidate moves for a position, evaluate each with Stockfish.
    Returns list of (token_ids, chosen_token_id, reward) for each sample.

    This is the key improvement from nanochat: instead of evaluating 1 move,
    we try N moves and learn which ones are better than average.
    """
    legal_moves = list(board.legal_moves)
    legal_sans = [board.san(m) for m in legal_moves]

    legal_ids = set()
    for san in legal_sans:
        if san in token_to_id:
            legal_ids.add(token_to_id[san])

    if not legal_ids:
        return []

    # Generate N samples from the model's distribution
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    next_logits = logits[0, -1, :]

    # Mask illegal moves
    mask = torch.full_like(next_logits, float("-inf"))
    for lid in legal_ids:
        mask[lid] = 0
    next_logits = next_logits + mask

    probs = F.softmax(next_logits / temperature, dim=-1)

    # Sample N moves (with replacement — same move can appear multiple times)
    num_samples = min(num_samples, len(legal_ids))
    sampled_ids = torch.multinomial(probs, num_samples, replacement=True)

    # Evaluate each sampled move with Stockfish
    experiences = []
    board_before = board.copy()

    for token_id in sampled_ids.tolist():
        san = id_to_token[token_id]
        move = board.parse_san(san)

        board.push(move)
        reward = evaluator.get_reward(board_before, board)
        board.pop()

        experiences.append((list(token_ids), token_id, reward))

    return experiences


def play_with_feedback_manual(model, token_to_id, id_to_token, device, human_color="black", temperature=0.8):
    """Interactive game with human feedback."""
    board = chess.Board()
    human_white = (human_color == "white")
    token_ids = [token_to_id[BOS_TOKEN]]
    if INFERENCE_ELO_BUCKET in token_to_id:
        token_ids.append(token_to_id[INFERENCE_ELO_BUCKET])
    experiences = []

    print(f"\n{'='*50}")
    print(f"  RLHF - Manual Feedback")
    print(f"  You play {'WHITE' if human_white else 'BLACK'}")
    print(f"  After each LLM move, rate it:")
    print(f"    [b] good  [m] bad  [enter] neutral")
    print(f"{'='*50}")

    while not board.is_game_over():
        render_board(board, perspective_white=human_white)

        is_white_turn = board.turn == chess.WHITE
        is_human_turn = (human_white and is_white_turn) or (not human_white and not is_white_turn)
        turn_label = "White" if is_white_turn else "Black"

        if is_human_turn:
            move = get_human_move(board, turn_label)
            if move is None:
                return experiences
            san = board.san(move)
        else:
            move = predict_chess_move(model, token_ids, token_to_id, id_to_token, board, device, temperature)
            san = board.san(move)
            print(f"  [{turn_label}] LLM plays: {san}")

            fb = input(f"  Rate [b]good / [m]bad / [enter] neutral: ").strip().lower()
            if fb == "b":
                reward = 1.0
            elif fb == "m":
                reward = -1.0
            else:
                reward = 0.0

            if san in token_to_id:
                experiences.append((list(token_ids), token_to_id[san], reward))

        if san in token_to_id:
            token_ids.append(token_to_id[san])
        board.push(move)

    result = board.result()
    render_board(board, perspective_white=human_white)
    print(f"\nResult: {result}")

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

    return experiences


def play_with_feedback_auto(model, token_to_id, id_to_token, device, evaluator, num_samples=8, temperature=1.0):
    """
    Automatic game with multi-sample evaluation.
    For each position, samples N moves and evaluates all of them.
    """
    board = chess.Board()
    token_ids = [token_to_id[BOS_TOKEN]]
    if INFERENCE_ELO_BUCKET in token_to_id:
        token_ids.append(token_to_id[INFERENCE_ELO_BUCKET])
    experiences = []
    moves_played = 0

    while not board.is_game_over() and moves_played < 200:
        # Sample multiple moves and evaluate each one
        position_exps = sample_multiple_moves(
            model, token_ids, token_to_id, id_to_token,
            board, evaluator, device,
            num_samples=num_samples, temperature=temperature,
        )
        experiences.extend(position_exps)

        # Actually play the best move (greedy for game progression)
        if position_exps:
            best_exp = max(position_exps, key=lambda e: e[2])
            best_token_id = best_exp[1]
            san = id_to_token[best_token_id]
        else:
            import random
            move = random.choice(list(board.legal_moves))
            san = board.san(move)

        if san in token_to_id:
            token_ids.append(token_to_id[san])
        board.push(board.parse_san(san))
        moves_played += 1

    result = board.result() if board.is_game_over() else "*"

    # Result bonus
    for i in range(len(experiences)):
        tid, tar, rew = experiences[i]
        if result == "1-0":
            bonus = 2.0
        elif result == "0-1":
            bonus = -2.0
        else:
            bonus = 0
        experiences[i] = (tid, tar, rew + bonus)

    return experiences, result


def run_rlhf(feedback="auto", n_games=50, rounds=5, lr=5e-5, temperature=0.8, selftrained=False, num_samples=8):
    """Main RLHF loop."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = get_model_path(selftrained=selftrained)
    model, token_to_id, id_to_token = load_model(vocab_path, model_path, device)
    vocab_size = len(token_to_id)

    evaluator = None
    if feedback == "auto":
        if find_stockfish():
            evaluator = StockfishEvaluator(depth=10, time_limit=0.05)
            print(f"Using Stockfish")
        else:
            print("Stockfish not found, using heuristic.")
            evaluator = HeuristicEvaluator()
    elif feedback == "heuristic":
        evaluator = HeuristicEvaluator()

    print(f"\n{'='*50}")
    print(f"  RLHF ({'Manual' if feedback == 'manual' else 'Automatic'})")
    print(f"  Rounds: {rounds}, Games/round: {n_games}")
    print(f"  Samples per position: {num_samples}")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    for round_num in range(1, rounds + 1):
        print(f"\n{'─'*50}")
        print(f"ROUND {round_num}/{rounds}")
        print(f"{'─'*50}")

        all_experiences = []

        if feedback == "manual":
            exps = play_with_feedback_manual(model, token_to_id, id_to_token, device, temperature=temperature)
            all_experiences.extend(exps)
        else:
            results = {}
            for _ in tqdm(range(n_games), desc="Playing"):
                exps, result = play_with_feedback_auto(
                    model, token_to_id, id_to_token, device, evaluator,
                    num_samples=num_samples, temperature=temperature,
                )
                all_experiences.extend(exps)
                results[result] = results.get(result, 0) + 1
            print(f"  Results: {results}")

        if all_experiences:
            rewards = [r for _, _, r in all_experiences]
            avg_reward = sum(rewards) / len(rewards)
            pos = sum(1 for r in rewards if r > 0)
            neg = sum(1 for r in rewards if r < 0)
            print(f"  Experiences: {len(all_experiences)} ({num_samples}x per position)")
            print(f"  Avg reward: {avg_reward:.3f}")
            print(f"  Positive: {pos}, Negative: {neg}")

        # On-policy: train on this round's fresh data, then discard
        print(f"\n  Training (REINFORCE with advantages)...")
        model = rlhf_train(model, all_experiences, vocab_size, device, lr=lr)

        checkpoint = {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "d_model": 128, "n_heads": 4, "n_layers": 4, "max_len": 256,
            "round": round_num, "training": f"rlhf_{feedback}",
            "epoch": round_num, "val_loss": 0,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, "best_model_rlhf.pt"))
        print(f"  Model saved")

    if evaluator:
        evaluator.close()

    print(f"\n{'='*50}")
    print(f"  RLHF COMPLETE")
    print(f"{'='*50}")
