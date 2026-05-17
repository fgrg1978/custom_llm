# LLM Factory

Platform for building domain-specialized LLMs. Currently: chess.

```
┌─────────────────────────────────────────────────────┐
│                    LLM Factory                       │
├──────────┬──────────────────────┬───────────────────┤
│  core/   │  domains/chess/      │  cloud/           │
│          │                      │                   │
│  Trans-  │  Tokenizer (PGN)     │  Setup / sync /   │
│  former  │  Evaluator (SF)      │  train / eval     │
│  Trainer │  Distill (SF)        │  for Lambda       │
│  RLHF    │  Play / Selfplay     │  Labs (A100)      │
│          │  Tournament / ELO    │                   │
└──────────┴──────────────────────┴───────────────────┘
```

**Current status (Phase 1 + distill):** ~692 ELO vs Stockfish (previous baseline: ~420 ELO). See `ROADMAP.md` for the full phase plan (final target ~2300 ELO).

## Structure

```
custom_llm/
├── core/                          # GENERIC (reusable)
│   ├── transformer.py             # Transformer decoder (GPT-like)
│   ├── dataset.py                 # Dataset + vocab
│   ├── trainer.py                 # Train loop (CUDA bf16 / MPS / CPU)
│   ├── generator.py               # Sampling + model loading
│   ├── rlhf.py                    # Generic policy gradient
│   └── experiments.py             # Per-run JSON tracking
│
├── domains/chess/                 # DOMAIN: chess
│   ├── tokenizer.py               # PGN → tokens (with ELO buckets)
│   ├── prepare.py                 # Download + parallel parsing (Lichess .zst)
│   ├── distill.py                 # Generate (position → best move) with Stockfish
│   ├── evaluator.py               # Stockfish / heuristic
│   ├── ui.py                      # ASCII board
│   ├── play.py                    # Human vs LLM
│   ├── selfplay.py                # LLM vs LLM + self-training
│   ├── rlhf.py                    # RLHF with feedback
│   ├── evaluate_elo.py            # Measures ELO against Stockfish L0-L5
│   ├── tournament.py              # Relative ELO between two checkpoints
│   ├── data/                      # Generated data (gitignored)
│   └── checkpoints/               # Trained models
│
├── cloud/                         # Cloud GPU training (Lambda Labs)
│   ├── setup.sh, sync.sh          # Provision + code sync
│   ├── train.sh, run_all.sh       # Full pipeline (prepare → train → eval)
│   ├── phase3_eval.sh, eval_only.sh
│   ├── download.sh, finalize.sh
│   └── sanity.py, gpu_bench.py
│
├── experiments/                   # Per-run tracking (JSON per run, gitignored)
├── ROADMAP.md                     # Phased plan (Phase 0 → 6)
├── cli.py                         # Unified CLI
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install stockfish      # macOS — required for evaluate, distill, RLHF
# apt-get install stockfish # Ubuntu/Debian
```

## Commands

```bash
source venv/bin/activate

# 1. Prepare dataset (Lichess, filtered by ELO)
python cli.py chess prepare --max-games 50000 --min-elo 1600

# 2. Train
python cli.py chess train --epochs 20
python cli.py chess train --epochs 30 --use-distill   # includes sequences_distill.pt

# 3. Distill from Stockfish (optional, improves tactical quality)
python cli.py chess distill --max-games 10000 --positions-per-game 5 --depth 10

# 4. Measure ELO
python cli.py chess evaluate --games 30 --levels 0 1 2 3 4

# 5. Tournament between two checkpoints
python cli.py chess tournament --a base --b rlhf --games 50

# 6. Play
python cli.py chess play --color white
python cli.py chess play --color white --temperature 1.0   # more active
python cli.py chess play --color white --temperature 0.3   # more conservative

# 7. LLM vs LLM
python cli.py chess selfplay --games 10 --verbose

# 8. RLHF (Stockfish evaluates and refines)
python cli.py chess rlhf --feedback auto --rounds 5 --games 50
```

## Cloud training (Lambda Labs A100)

Full end-to-end pipeline (~30-60 min on A100, ~$1-2):

```bash
# Locally: start A100 instance on Lambda Labs, copy IP
export SSH_KEY=~/.ssh/lambda.pem

bash cloud/sync.sh ubuntu@<IP>          # uploads the code
ssh -i $SSH_KEY ubuntu@<IP> 'bash custom_llm/cloud/setup.sh'
ssh -i $SSH_KEY ubuntu@<IP> 'cd custom_llm && nohup bash cloud/run_all.sh > /tmp/run.log 2>&1 &'

# When done:
bash cloud/download.sh ubuntu@<IP>      # downloads best_model.pt, vocab, RESULT.txt
# Terminate instance from the Lambda Labs web UI
```

## Move notation (when playing)

| Action              | Input    |
|---------------------|----------|
| Pawn to e4          | `e4`     |
| Knight to f3        | `Nf3`    |
| Bishop takes e5     | `Bxe5`   |
| Short castle        | `O-O`    |
| Long castle         | `O-O-O`  |
| Promote to queen    | `e8=Q`   |
| Resign game         | `salir`  |

## Saved models

```
domains/chess/checkpoints/
├── best_model.pt              # best checkpoint (Phase 1 + distill, 692 ELO)
├── best_model_rlhf.pt         # after RLHF (optional)
├── best_model_selftrained.pt  # after self-play training (optional)
└── RESULT.txt                 # last evaluation gate
```

## Training pipeline (summary)

```
Phase 0: Baseline                  -> Transformer 4×128, no filters (~420 ELO)
Phase 1: Scaled imitation          -> 6×256, dataset ≥1600 ELO, ELO buckets  (~692 ELO)
Phase 3: Distillation (Stockfish)  -> Positions labeled with best move
Phase 2: Board encoder             -> FEN as input (next target)
Phase 5: Intensive RLHF            -> Reward shaping with Stockfish
Phase 6: Thinking                  -> `<think>...</think>` tokens
```

Details in `ROADMAP.md`.
