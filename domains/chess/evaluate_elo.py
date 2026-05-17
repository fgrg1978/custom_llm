"""
Estima la ELO del LLM de ajedrez jugando contra oponentes de distinta fuerza:
un jugador aleatorio (L0) como ancla inferior, y Stockfish debilitado (L1-L5)
con Skill Level + depth como niveles calibrados.

Agrega la ELO final solo sobre niveles informativos (score estricto en (0,1));
los niveles saturados (pierde todas / gana todas) dan cotas, no estimaciones
puntuales, y se reportan como upper/lower bound.
"""

import os
import math
import random
import chess
import chess.engine
import torch
from tqdm import tqdm

from core.dataset import BOS_TOKEN
from core.generator import load_model
from domains.chess.tokenizer import ELO_BUCKETS

INFERENCE_ELO_BUCKET = ELO_BUCKETS[-2][1]  # <ELO_2200> — nivel alto de los datos de entrenamiento
from domains.chess.play import predict_chess_move, get_model_path
from domains.chess.evaluator import find_stockfish

DOMAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DOMAIN_DIR, "data")

# Perfiles de oponente. L0 es un jugador aleatorio (ancla ~200 ELO). L1-L5
# usan Stockfish con Skill Level + depth para aproximar rangos de ELO. Los
# valores de ELO son estimaciones de la comunidad; lo relevante es que la
# curva sea monotona y comparable entre checkpoints.
OPPONENTS = [
    {"name": "L0", "elo": 200,  "type": "random"},
    {"name": "L1", "elo": 800,  "type": "stockfish", "skill": 0,  "depth": 1},
    {"name": "L2", "elo": 1100, "type": "stockfish", "skill": 2,  "depth": 2},
    {"name": "L3", "elo": 1400, "type": "stockfish", "skill": 5,  "depth": 3},
    {"name": "L4", "elo": 1700, "type": "stockfish", "skill": 10, "depth": 5},
    {"name": "L5", "elo": 2000, "type": "stockfish", "skill": 15, "depth": 8},
]


def opponent_move(board, engine, opp):
    """Devuelve la jugada del oponente segun su tipo."""
    if opp["type"] == "random":
        return random.choice(list(board.legal_moves))
    return engine.play(board, chess.engine.Limit(depth=opp["depth"])).move


def play_game(model, token_to_id, id_to_token, device, engine, opp,
              llm_is_white, temperature, max_moves=200):
    """Juega una partida LLM vs oponente. Devuelve score del LLM (1/0.5/0)."""
    board = chess.Board()
    token_ids = [token_to_id[BOS_TOKEN]]
    if INFERENCE_ELO_BUCKET in token_to_id:
        token_ids.append(token_to_id[INFERENCE_ELO_BUCKET])
    moves = 0

    while not board.is_game_over() and moves < max_moves:
        is_llm_turn = (board.turn == chess.WHITE) == llm_is_white
        if is_llm_turn:
            move = predict_chess_move(
                model, token_ids, token_to_id, id_to_token,
                board, device, temperature,
            )
        else:
            move = opponent_move(board, engine, opp)

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
        return 1.0 if llm_is_white else 0.0
    return 0.0 if llm_is_white else 1.0


def estimate_elo(opp_elo, score, n_games):
    """
    ELO a partir del score vs oponente de ELO conocida (formula logistica).
    Devuelve (elo, lo_95, hi_95, saturated).
    Saturated = True si score in {0, 1} (la estimacion es solo cota).
    """
    if score <= 0:
        return opp_elo - 400, opp_elo - 600, opp_elo - 200, True
    if score >= 1:
        return opp_elo + 400, opp_elo + 200, opp_elo + 600, True
    delta = 400 * math.log10(score / (1 - score))
    elo = opp_elo + delta
    se = math.sqrt(score * (1 - score) / n_games)
    s_lo = max(0.01, score - 1.96 * se)
    s_hi = min(0.99, score + 1.96 * se)
    lo = opp_elo + 400 * math.log10(s_lo / (1 - s_lo))
    hi = opp_elo + 400 * math.log10(s_hi / (1 - s_hi))
    return elo, lo, hi, False


def aggregate_elo(results):
    """
    Agrega resultados evitando el sesgo de niveles saturados.
    - Si hay >=1 nivel informativo (score estricto en (0,1)): media ponderada por n_games.
    - Si pierde todas en todos los niveles: devuelve cota superior (weakest_opp - 200).
    - Si gana todas en todos los niveles: devuelve cota inferior (strongest_opp + 200).
    Devuelve dict con keys: estimate, kind, detail.
    """
    informative = [r for r in results if not r["saturated"]]
    lost_all = [r for r in results if r["saturated"] and r["score_pct"] <= 0]
    won_all = [r for r in results if r["saturated"] and r["score_pct"] >= 1]

    if informative:
        total = sum(r["games"] for r in informative)
        est = sum(r["elo"] * r["games"] for r in informative) / total
        levels = ", ".join(r["name"] for r in informative)
        return {
            "estimate": est,
            "kind": "point",
            "detail": f"media ponderada de niveles informativos ({levels})",
        }

    if lost_all:
        weakest = min(lost_all, key=lambda r: r["opp_elo"])
        return {
            "estimate": weakest["opp_elo"] - 200,
            "kind": "upper_bound",
            "detail": f"pierde todo vs {weakest['name']} (~{weakest['opp_elo']}); ELO esta por debajo",
        }

    if won_all:
        strongest = max(won_all, key=lambda r: r["opp_elo"])
        return {
            "estimate": strongest["opp_elo"] + 200,
            "kind": "lower_bound",
            "detail": f"gana todo vs {strongest['name']} (~{strongest['opp_elo']}); ELO esta por encima",
        }

    return {"estimate": 0.0, "kind": "none", "detail": "sin datos"}


def evaluate_elo(selftrained=False, rlhf=False, n_games=10, levels=None,
                 temperature=0.3, seed=42):
    """Loop principal de evaluacion. Determinista si se pasa seed."""
    if levels is None:
        levels = [0, 1, 2, 3, 4]  # L0 (random) hasta L4; L5 opcional

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    model_path = get_model_path(selftrained=selftrained, rlhf=rlhf)
    model, token_to_id, id_to_token = load_model(vocab_path, model_path, device)

    sf_path = find_stockfish()

    print(f"\n{'='*65}")
    print(f"  Evaluacion de ELO")
    print(f"  Modelo:         {os.path.basename(model_path)}")
    print(f"  Partidas/nivel: {n_games}")
    print(f"  Temperatura:    {temperature}")
    print(f"  Seed:           {seed}")
    print(f"  Niveles:        {levels}")
    print(f"{'='*65}\n")

    results = []

    for level_idx in levels:
        if not (0 <= level_idx < len(OPPONENTS)):
            print(f"  Nivel invalido: {level_idx}, omitido")
            continue
        opp = OPPONENTS[level_idx]

        if opp["type"] == "stockfish":
            if sf_path is None:
                print(f"  Stockfish no encontrado, omitiendo {opp['name']}")
                continue
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            engine.configure({"Skill Level": opp["skill"]})
            desc = f"{opp['name']} ~{opp['elo']} ELO (SF skill={opp['skill']}, d={opp['depth']})"
        else:
            engine = None
            desc = f"{opp['name']} ~{opp['elo']} ELO (random)"

        wins = draws = losses = 0
        score = 0.0

        pbar = tqdm(range(n_games), desc=desc)
        for i in pbar:
            llm_is_white = (i % 2 == 0)  # alternar colores
            r = play_game(model, token_to_id, id_to_token, device, engine, opp,
                          llm_is_white, temperature)
            score += r
            if r == 1.0:
                wins += 1
            elif r == 0.5:
                draws += 1
            else:
                losses += 1
            pbar.set_postfix(W=wins, D=draws, L=losses)

        if engine is not None:
            engine.quit()

        score_pct = score / n_games
        est_elo, lo, hi, saturated = estimate_elo(opp["elo"], score_pct, n_games)

        results.append({
            "name": opp["name"], "opp_elo": opp["elo"],
            "wins": wins, "draws": draws, "losses": losses,
            "games": n_games, "score_pct": score_pct,
            "elo": est_elo, "lo": lo, "hi": hi, "saturated": saturated,
        })

    agg = aggregate_elo(results)

    print(f"\n{'='*65}")
    print(f"  Resumen")
    print(f"{'='*65}")
    print(f"  {'Opp':<4} {'ELOopp':>7} {'W':>3} {'D':>3} {'L':>3}"
          f" {'Score':>7} {'EstELO':>8} {'95%CI':>15} {'Info':>5}")
    for r in results:
        ci = f"[{r['lo']:.0f}, {r['hi']:.0f}]"
        info = "sat" if r["saturated"] else "ok"
        print(f"  {r['name']:<4} {r['opp_elo']:>7} {r['wins']:>3} {r['draws']:>3}"
              f" {r['losses']:>3} {r['score_pct']:>6.1%} {r['elo']:>8.0f} {ci:>15} {info:>5}")

    prefix = {
        "point": "Estimacion puntual",
        "upper_bound": "Cota superior",
        "lower_bound": "Cota inferior",
        "none": "Sin datos",
    }[agg["kind"]]
    print(f"\n  {prefix}: ~{agg['estimate']:.0f} ELO")
    print(f"  ({agg['detail']})")
    print(f"{'='*65}\n")

    return {"estimate": agg["estimate"], "kind": agg["kind"], "results": results}
