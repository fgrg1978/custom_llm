"""
CLI unificado para LLM Factory.

Uso:
  python cli.py chess prepare --max-games 50000
  python cli.py chess train --epochs 20
  python cli.py chess play --color white
  python cli.py chess selfplay --games 10
  python cli.py chess selftrain --rounds 10
  python cli.py chess rlhf --feedback auto
  python cli.py chess evaluate --games 10
  python cli.py chess tournament --a base --b rlhf --games 50
"""

import argparse
import sys
import warnings
import os

# Silence PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    parser = argparse.ArgumentParser(description="LLM Factory")
    parser.add_argument("domain", choices=["chess"], help="Dominio")
    parser.add_argument("action", choices=["prepare", "distill", "train", "play", "selfplay", "selftrain", "rlhf", "evaluate", "tournament"],
                        help="Accion a ejecutar")

    # Argumentos opcionales comunes
    parser.add_argument("--max-games", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping: stop if val_loss does not improve for N epochs")
    parser.add_argument("--min-elo", type=int, default=1600,
                        help="Prepare: filtro minimo de ELO (ambos jugadores)")
    parser.add_argument("--url", type=str, default=None,
                        help="Prepare/Distill: URL del .zst de Lichess (default: 2022-11)")
    parser.add_argument("--positions-per-game", type=int, default=5,
                        help="Distill: posiciones a etiquetar por partida")
    parser.add_argument("--depth", type=int, default=10,
                        help="Distill: profundidad de Stockfish")
    parser.add_argument("--workers", type=int, default=None,
                        help="Distill: numero de Stockfish paralelos (default: cpu_count-2)")
    parser.add_argument("--use-distill", action="store_true",
                        help="Train: incluir sequences_distill.pt si existe")
    parser.add_argument("--no-autocast", action="store_true",
                        help="Train: desactivar mixed precision (float16)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Train: steps de warmup para el scheduler cosine")
    parser.add_argument("--color", choices=["white", "black"], default="white")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--feedback", choices=["manual", "auto", "heuristic"], default="auto")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="RLHF: number of candidate moves per position (like nanochat)")
    parser.add_argument("--levels", type=int, nargs="+", default=None,
                        help="Evaluate: niveles (0=random, 1-5=Stockfish). Default: 0 1 2 3 4")
    parser.add_argument("--a", type=str, default="base",
                        help="Tournament: checkpoint A (base/selftrained/rlhf o path)")
    parser.add_argument("--b", type=str, default="rlhf",
                        help="Tournament: checkpoint B (base/selftrained/rlhf o path)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed para evaluacion determinista")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--selftrained", action="store_true")
    parser.add_argument("--rlhf", action="store_true")

    args = parser.parse_args()

    if args.domain == "chess":
        run_chess(args)


def run_chess(args):
    import os

    domain_dir = os.path.join(os.path.dirname(__file__), "domains", "chess")
    data_dir = os.path.join(domain_dir, "data")
    checkpoints_dir = os.path.join(domain_dir, "checkpoints")

    if args.action == "prepare":
        from domains.chess.prepare import prepare, DEFAULT_URL
        url = args.url if args.url else DEFAULT_URL
        prepare(max_games=args.max_games, min_elo=args.min_elo, url=url)

    elif args.action == "distill":
        from domains.chess.distill import distill, DEFAULT_URL as DISTILL_URL
        url = args.url if args.url else DISTILL_URL
        distill(
            vocab_path=os.path.join(data_dir, "vocab.json"),
            n_games=args.max_games,
            positions_per_game=args.positions_per_game,
            min_elo=args.min_elo,
            depth=args.depth,
            n_workers=args.workers,
            url=url,
        )

    elif args.action == "train":
        from core.trainer import train
        extra_path = os.path.join(data_dir, "sequences_distill.pt") if args.use_distill else None
        train(
            vocab_path=os.path.join(data_dir, "vocab.json"),
            data_path=os.path.join(data_dir, "sequences.pt"),
            checkpoints_dir=checkpoints_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_len=args.max_len,
            patience=args.patience,
            autocast=not args.no_autocast,
            warmup_steps=args.warmup_steps,
            extra_data_path=extra_path,
        )

    elif args.action == "play":
        from domains.chess.play import play
        play(color=args.color, temperature=args.temperature,
             selftrained=args.selftrained, rlhf=args.rlhf)

    elif args.action == "selfplay":
        from domains.chess.selfplay import run_selfplay
        run_selfplay(n_games=args.games, temperature=args.temperature, verbose=args.verbose)

    elif args.action == "selftrain":
        from domains.chess.selfplay import selftrain
        selftrain(rounds=args.rounds, games_per_round=args.games,
                  lr=args.lr, temperature=args.temperature)

    elif args.action == "rlhf":
        from domains.chess.rlhf import run_rlhf
        run_rlhf(feedback=args.feedback, n_games=args.games, rounds=args.rounds,
                 lr=args.lr, temperature=args.temperature, selftrained=args.selftrained,
                 num_samples=args.num_samples)

    elif args.action == "evaluate":
        from domains.chess.evaluate_elo import evaluate_elo
        from core.experiments import log_experiment
        result = evaluate_elo(selftrained=args.selftrained, rlhf=args.rlhf,
                              n_games=args.games, levels=args.levels,
                              temperature=args.temperature, seed=args.seed)
        log_experiment("evaluate", vars(args), result)

    elif args.action == "tournament":
        from domains.chess.tournament import tournament
        from core.experiments import log_experiment
        result = tournament(a=args.a, b=args.b, n_games=args.games,
                            temperature=args.temperature, seed=args.seed)
        log_experiment("tournament", vars(args), result)

    print_next_steps(args.action)


def print_next_steps(current_action):
    """Muestra los comandos disponibles al finalizar."""
    steps = {
        "prepare": {
            "done": "Datos preparados",
            "next": [
                ("Entrenar modelo base", "python cli.py chess train --epochs 20"),
                ("Entrenar rapido", "python cli.py chess train --epochs 5"),
            ],
        },
        "train": {
            "done": "Modelo entrenado",
            "next": [
                ("Medir ELO (baseline)", "python cli.py chess evaluate --games 10"),
                ("Jugar como blancas", "python cli.py chess play --color white"),
                ("Jugar como negras", "python cli.py chess play --color black"),
                ("LLM vs LLM", "python cli.py chess selfplay --games 10 --verbose"),
            ],
        },
        "evaluate": {
            "done": "Evaluacion de ELO completada",
            "next": [
                ("Evaluar modelo self-trained", "python cli.py chess evaluate --games 10 --selftrained"),
                ("Evaluar modelo RLHF", "python cli.py chess evaluate --games 10 --rlhf"),
                ("Torneo base vs RLHF", "python cli.py chess tournament --a base --b rlhf --games 50"),
                ("Jugar contra el LLM", "python cli.py chess play --color white"),
            ],
        },
        "tournament": {
            "done": "Torneo completado",
            "next": [
                ("Medir ELO absoluta", "python cli.py chess evaluate --games 10"),
                ("Torneo base vs selftrained", "python cli.py chess tournament --a base --b selftrained --games 50"),
            ],
        },
        "play": {
            "done": "Partida finalizada",
            "next": [
                ("Jugar otra vez", "python cli.py chess play --color white"),
                ("Cambiar temperatura", "python cli.py chess play --color white --temperature 0.3"),
                ("LLM vs LLM", "python cli.py chess selfplay --games 10 --verbose"),
                ("Auto-entrenar", "python cli.py chess selftrain --rounds 10 --games 100"),
            ],
        },
        "selfplay": {
            "done": "Self-play completado",
            "next": [
                ("Auto-entrenar", "python cli.py chess selftrain --rounds 10 --games 100"),
                ("RLHF con Stockfish", "python cli.py chess rlhf --feedback auto --rounds 5 --games 50"),
                ("Jugar contra el LLM", "python cli.py chess play --color white"),
            ],
        },
        "selftrain": {
            "done": "Auto-entrenamiento completado",
            "next": [
                ("Jugar vs modelo mejorado", "python cli.py chess play --color white --selftrained"),
                ("RLHF con Stockfish", "python cli.py chess rlhf --feedback auto --rounds 5 --games 50"),
                ("Ver LLM vs LLM", "python cli.py chess selfplay --games 10 --verbose"),
            ],
        },
        "rlhf": {
            "done": "RLHF completado",
            "next": [
                ("Jugar vs modelo RLHF", "python cli.py chess play --color white --rlhf"),
                ("Jugar vs modelo base", "python cli.py chess play --color white"),
                ("Mas RLHF", "python cli.py chess rlhf --feedback auto --rounds 5 --games 50"),
            ],
        },
    }

    info = steps.get(current_action)
    if not info:
        return

    print(f"\n{'='*55}")
    print(f"  {info['done']}. Siguientes pasos:")
    print(f"{'='*55}")
    for desc, cmd in info["next"]:
        print(f"\n  {desc}:")
        print(f"    {cmd}")
    print()


if __name__ == "__main__":
    main()
