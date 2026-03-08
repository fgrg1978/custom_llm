"""
Generador generico: carga modelo y genera tokens con sampling.
"""

import os
import torch
import torch.nn.functional as F

from core.transformer import SequenceTransformer
from core.dataset import load_vocab


def load_model(vocab_path, checkpoint_path, device):
    """Carga modelo y vocabulario."""
    token_to_id, id_to_token = load_vocab(vocab_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SequenceTransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        max_len=checkpoint["max_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    info = f"epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', 0):.4f}"
    print(f"Modelo cargado ({info})")
    return model, token_to_id, id_to_token


def predict_next_token(model, token_ids, valid_token_ids, device, temperature=0.8):
    """
    Predice el siguiente token dado un historial.

    Args:
        model: el transformer
        token_ids: lista de token IDs hasta ahora
        valid_token_ids: set de token IDs validos (para filtrar)
        device: dispositivo
        temperature: temperatura de sampling

    Returns:
        token_id seleccionado
    """
    if not valid_token_ids:
        return None

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)

    next_logits = logits[0, -1, :]

    # Mascara: solo tokens validos
    mask = torch.full_like(next_logits, float("-inf"))
    for tid in valid_token_ids:
        mask[tid] = 0
    next_logits = next_logits + mask

    probs = F.softmax(next_logits / temperature, dim=-1)
    token_id = torch.multinomial(probs, 1).item()
    return token_id
