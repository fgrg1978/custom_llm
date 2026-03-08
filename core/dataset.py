"""
Dataset generico para secuencias de tokens.
Cada dominio provee sus propias secuencias tokenizadas.
"""

import json
import torch
from torch.utils.data import Dataset


# Tokens especiales (comunes a todos los dominios)
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


def save_vocab(token_to_id, path):
    """Guarda el vocabulario en un archivo JSON."""
    with open(path, "w") as f:
        json.dump(token_to_id, f, indent=2)


def load_vocab(path):
    """Carga el vocabulario desde un archivo JSON."""
    with open(path) as f:
        token_to_id = json.load(f)
    id_to_token = {int(i): token for token, i in token_to_id.items()}
    return token_to_id, id_to_token


class SequenceDataset(Dataset):
    """Dataset generico de secuencias de tokens para next-token prediction."""

    def __init__(self, sequences, max_len=256):
        self.max_len = max_len
        self.sequences = [s[:max_len] for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Input: todos los tokens menos el ultimo
        # Target: todos los tokens menos el primero (shifted right)
        x = seq[:-1]
        y = seq[1:]

        # Padding
        pad_len = self.max_len - 1 - len(x)
        x = x + [0] * pad_len
        y = y + [0] * pad_len

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
