"""
CLI unificado para LLM Factory.

Uso:
  python cli.py chess prepare --max-games 50000
  python cli.py chess train --epochs 20
  python cli.py chess play --color white
  python cli.py chess selfplay --games 10
  python cli.py chess selftrain --rounds 10
  python cli.py chess rlhf --feedback auto
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="LLM Factory")
    parser.add_argument("domain", choices=["chess"], help="Dominio")
    parser.add_argument("action", choices=["prepare", "train", "play", "selfplay", "selftrain", "rlhf"],
                        help="Accion a ejecutar")

    # Argumentos opcionales comunes
    parser.add_argument("--max-games", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--color", choices=["white", "black"], default="white")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--feedback", choices=["manual", "auto", "heuristic"], default="auto")
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
        from domains.chess.prepare import prepare
        prepare(max_games=args.max_games)

    elif args.action == "train":
        from core.trainer import train
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
                 lr=args.lr, temperature=args.temperature, selftrained=args.selftrained)

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
                ("Jugar como blancas", "python cli.py chess play --color white"),
                ("Jugar como negras", "python cli.py chess play --color black"),
                ("LLM vs LLM", "python cli.py chess selfplay --games 10 --verbose"),
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
