"""
LLM vs LLM y auto-entrenamiento para ajedrez.
"""

import os
import chess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import BOS_TOKEN, EOS_TOKEN, SequenceDataset
from core.generator import load_model, predict_next_token
from domains.chess.play import predict_chess_move, get_model_path
from domains.chess.tokenizer import RESULT_MAP
from domains.chess.ui import render_board

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DOMAIN_DIR, "checkpoints")


def selfplay_game(model, token_to_id, id_to_token, device, temperature=0.8, max_moves=200, verbose=True):
    """Una partida completa LLM vs LLM."""
    board = chess.Board()
    token_ids = [token_to_id[BOS_TOKEN]]
    moves_played = []

    while not board.is_game_over() and len(moves_played) < max_moves:
        move = predict_chess_move(model, token_ids, token_to_id, id_to_token, board, device, temperature)
        san = board.san(move)
        moves_played.append(san)

        if san in token_to_id:
            token_ids.append(token_to_id[san])
        board.push(move)

        if verbose and len(moves_played) % 10 == 0:
            print(f"  Movimiento {len(moves_played)}...")

    result = board.result() if board.is_game_over() else "*"

    if verbose:
        render_board(board)
        print(f"Movimientos: {len(moves_played)}")
        print(f"Resultado: {result}")
        print(f"Partida: {' '.join(moves_played)}")

    return result, moves_played, token_ids


def run_selfplay(n_games=10, temperature=0.8, verbose=False):
    """Ejecuta multiples partidas LLM vs LLM."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = get_model_path()
    model, token_to_id, id_to_token = load_model(vocab_path, model_path, device)

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    all_games = []

    print(f"\n{'='*40}")
    print(f"  LLM vs LLM - {n_games} partidas")
    print(f"  Temperatura: {temperature}")
    print(f"{'='*40}\n")

    for i in tqdm(range(n_games), desc="Partidas"):
        if verbose:
            print(f"\n--- Partida {i+1}/{n_games} ---")

        result, moves, _ = selfplay_game(
            model, token_to_id, id_to_token, device,
            temperature=temperature, verbose=verbose,
        )
        results[result] = results.get(result, 0) + 1
        all_games.append({"result": result, "moves": moves, "n_moves": len(moves)})

    print(f"\n{'='*40}")
    print(f"  RESULTADOS")
    print(f"{'='*40}")
    print(f"  Blancas ganan: {results.get('1-0', 0)}")
    print(f"  Negras ganan:  {results.get('0-1', 0)}")
    print(f"  Empates:       {results.get('1/2-1/2', 0)}")
    print(f"  Inconclusas:   {results.get('*', 0)}")

    avg_moves = sum(g["n_moves"] for g in all_games) / len(all_games) if all_games else 0
    print(f"  Promedio movimientos: {avg_moves:.1f}")

    if all_games:
        longest = max(all_games, key=lambda g: g["n_moves"])
        print(f"\n  Partida mas larga ({longest['n_moves']} mov): {longest['result']}")
        print(f"  {' '.join(longest['moves'][:20])}...")


def selftrain(rounds=10, games_per_round=100, finetune_epochs=3, lr=1e-4, temperature=0.8):
    """Auto-entrenamiento: juega contra si mismo y aprende."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = get_model_path()
    model, token_to_id, id_to_token = load_model(vocab_path, model_path, device)

    vocab_size = len(token_to_id)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  AUTO-ENTRENAMIENTO")
    print(f"  Rondas: {rounds}")
    print(f"  Partidas por ronda: {games_per_round}")
    print(f"  Dispositivo: {device}")
    print(f"{'='*50}\n")

    best_decisive = 0

    for round_num in range(1, rounds + 1):
        print(f"\n{'─'*50}")
        print(f"RONDA {round_num}/{rounds}")
        print(f"{'─'*50}")

        # 1. Generar partidas
        print(f"\n1. Generando {games_per_round} partidas...")
        games = {"1-0": [], "0-1": [], "1/2-1/2": [], "*": []}

        for _ in tqdm(range(games_per_round), desc="Jugando"):
            result, moves, token_ids = selfplay_game(
                model, token_to_id, id_to_token, device,
                temperature=temperature, verbose=False,
            )
            if result in RESULT_MAP:
                token_ids.append(token_to_id[RESULT_MAP[result]])
            token_ids.append(token_to_id[EOS_TOKEN])
            games[result].append(token_ids)

        stats = {k: len(v) for k, v in games.items()}
        print(f"   Resultados: {stats}")

        # 2. Construir datos (partidas decisivas x3, empates x1)
        sequences = []
        for seq in games.get("1-0", []) + games.get("0-1", []):
            for _ in range(3):
                sequences.append(seq)
        for seq in games.get("1/2-1/2", []):
            sequences.append(seq)

        print(f"   Secuencias de entrenamiento: {len(sequences)}")

        # 3. Fine-tune
        if sequences:
            print(f"\n2. Fine-tuning ({finetune_epochs} epochs)...")
            dataset = SequenceDataset(sequences, max_len=256)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss(ignore_index=0)

            for epoch in range(finetune_epochs):
                epoch_loss = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                print(f"   Epoch {epoch+1}, loss: {epoch_loss/len(loader):.4f}")

        # 4. Guardar
        decisive = stats.get("1-0", 0) + stats.get("0-1", 0)
        decisive_pct = decisive / games_per_round * 100
        print(f"\n   Partidas decisivas: {decisive_pct:.1f}%")

        if decisive_pct >= best_decisive:
            best_decisive = decisive_pct
            checkpoint = {
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "d_model": 128, "n_heads": 4, "n_layers": 4, "max_len": 256,
                "round": round_num, "decisive_pct": decisive_pct,
                "epoch": round_num, "val_loss": 0,
            }
            torch.save(checkpoint, os.path.join(CHECKPOINTS_DIR, "best_model_selftrained.pt"))
            print(f"   -> Modelo guardado")

        temperature = max(0.3, temperature * 0.95)

    print(f"\n{'='*50}")
    print(f"  AUTO-ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor % decisivas: {best_decisive:.1f}%")
    print(f"{'='*50}")
