"""
Entrenamiento generico para cualquier dominio.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from core.transformer import SequenceTransformer
from core.dataset import SequenceDataset, load_vocab


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
):
    device = get_device()
    print(f"Dispositivo: {device}")

    # Cargar datos
    token_to_id, id_to_token = load_vocab(vocab_path)
    sequences = torch.load(data_path, weights_only=False)

    vocab_size = len(token_to_id)
    print(f"Vocabulario: {vocab_size} tokens")
    print(f"Secuencias: {len(sequences)}")

    # Dataset y split
    dataset = SequenceDataset(sequences, max_len=max_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

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

    os.makedirs(checkpoints_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            print(f"  -> Mejor modelo guardado (val_loss={val_loss:.4f})")

    print(f"\nEntrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
