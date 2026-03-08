# LLM Factory

Platform for building domain-specific LLMs. Currently supports chess.

```
┌─────────────────────────────────────┐
│           LLM Factory               │
├──────────┬──────────────────────────┤
│          │                          │
│  core/   │  domains/chess/          │
│          │                          │
│  Trans-  │  Tokenizer (PGN)        │
│  former  │  Evaluator (Stockfish)  │
│  Trainer │  UI (ASCII board)       │
│  RLHF    │  Play / Selfplay        │
│          │                          │
└──────────┴──────────────────────────┘
```

## Structure

```
custom_llm/
├── core/                          # GENERIC (reusable)
│   ├── transformer.py             # Transformer decoder (GPT-like)
│   ├── dataset.py                 # Dataset + vocabulary
│   ├── trainer.py                 # Training loop
│   ├── generator.py               # Sampling + model loading
│   └── rlhf.py                    # Generic policy gradient
│
├── domains/
│   └── chess/                     # DOMAIN: chess
│       ├── tokenizer.py           # PGN -> tokens
│       ├── evaluator.py           # Stockfish / heuristic
│       ├── ui.py                  # ASCII board
│       ├── prepare.py             # Download Lichess data
│       ├── play.py                # Human vs LLM
│       ├── selfplay.py            # LLM vs LLM + self-training
│       ├── rlhf.py                # RLHF with feedback
│       ├── data/                  # Data (generated)
│       └── checkpoints/           # Models (generated)
│
├── cli.py                         # Unified CLI
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install stockfish    # optional, for RLHF
```

## Commands

```bash
source venv/bin/activate

# Prepare data
python cli.py chess prepare --max-games 50000

# Train
python cli.py chess train --epochs 20
python cli.py chess train --epochs 5                    # quick

# Play
python cli.py chess play --color white
python cli.py chess play --color black
python cli.py chess play --color white --temperature 0.3    # conservative
python cli.py chess play --color white --temperature 1.5    # creative

# LLM vs LLM
python cli.py chess selfplay --games 10 --verbose

# Self-training
python cli.py chess selftrain --rounds 10 --games 100

# RLHF
python cli.py chess rlhf --feedback auto --rounds 5 --games 50
python cli.py chess rlhf --feedback manual --rounds 5
python cli.py chess rlhf --feedback heuristic --rounds 5 --games 50

# Play against improved models
python cli.py chess play --color white --selftrained
python cli.py chess play --color white --rlhf
```

## Saved models

```
domains/chess/checkpoints/
├── best_model.pt              # base (human imitation)
├── best_model_selftrained.pt  # self-trained
└── best_model_rlhf.pt         # RLHF (Stockfish)
```

## Move notation

| Action              | Input    |
|---------------------|----------|
| Pawn to e4          | `e4`     |
| Knight to f3        | `Nf3`   |
| Bishop captures e5  | `Bxe5`  |
| Kingside castle     | `O-O`   |
| Queenside castle    | `O-O-O` |
| Promote to queen    | `e8=Q`  |
| Quit game           | `salir` |

## Training pipeline

```
Phase 1: IMITATION (train)       -> learns from 50k human games
Phase 2: SELF-PLAY (selftrain)   -> plays itself and improves
Phase 3: RLHF (rlhf)            -> Stockfish evaluates and refines
```
