"""
Entrenamiento generico para cualquier dominio.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from core.transformer import SequenceTransformer
from core.dataset import SequenceDataset, load_vocab


def get_device():
    """Prefiere CUDA > MPS > CPU. CUDA es siempre mas rapido cuando disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device):
    """bf16 en CUDA (Ampere+), fp16 en MPS. bf16 evita gradient scaler y NaNs."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float16


def cosine_warmup_lambda(warmup_steps, total_steps, min_ratio=0.1):
    """LR schedule: linear warmup + cosine decay hasta min_ratio * lr."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def train(
    vocab_path,
    data_path,
    checkpoints_dir,
    epochs=20,
    batch_size=64,
    lr=3e-4,
    d_model=128,
    n_heads=4,
    n_layers=4,
    max_len=256,
    patience=5,
    autocast=True,
    warmup_steps=500,
    extra_data_path=None,
):
    device = get_device()
    print(f"Dispositivo: {device}")

    # Cargar datos (imitacion humana + opcionalmente distillation Stockfish)
    token_to_id, id_to_token = load_vocab(vocab_path)
    sequences = torch.load(data_path, weights_only=False)
    if extra_data_path and os.path.exists(extra_data_path):
        extra = torch.load(extra_data_path, weights_only=False)
        print(f"Datos extra cargados: {len(extra)} secuencias desde {extra_data_path}")
        sequences = sequences + extra

    vocab_size = len(token_to_id)
    print(f"Vocabulario: {vocab_size} tokens")
    print(f"Secuencias: {len(sequences)}")

    # Dataset y split
    dataset = SequenceDataset(sequences, max_len=max_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=2, persistent_workers=True, pin_memory=True)

    print(f"Train: {train_size}, Val: {val_size}")

    # Modelo
    model = SequenceTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {total_params:,}")

    # Entrenamiento
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_steps = epochs * len(train_loader)
    effective_warmup = min(warmup_steps, total_steps // 20)
    scheduler = LambdaLR(optimizer, cosine_warmup_lambda(effective_warmup, total_steps))

    use_autocast = autocast and device.type in ("mps", "cuda")
    ac_dtype = autocast_dtype(device)
    if use_autocast:
        print(f"Mixed precision: autocast {ac_dtype} en {device.type}")

    os.makedirs(checkpoints_dir, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Scheduler: cosine con {effective_warmup} steps de warmup sobre {total_steps} totales")
    print(f"Early stopping: patience={patience} (para si val_loss no mejora en {patience} epochs)")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            x, y = x.to(device), y.to(device)

            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=ac_dtype):
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=ac_dtype):
                        logits = model(x)
                        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                else:
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", end="")

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "max_len": max_len,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }
            path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save(checkpoint, path)
            print(f"  -> BEST (saved)")
        else:
            epochs_without_improvement += 1
            print(f"  (no improvement {epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. val_loss did not improve for {patience} epochs.")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
