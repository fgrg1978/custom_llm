"""
RLHF generico: fine-tune con policy gradient.
Cada dominio provee su propio evaluador y loop de interaccion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class RLHFDataset(Dataset):
    """Dataset con rewards para RLHF."""

    def __init__(self, experiences, max_len=256):
        """
        experiences: lista de (token_ids_hasta_aqui, token_jugado, reward)
        """
        self.data = []
        for token_ids, target_id, reward in experiences:
            if len(token_ids) >= max_len - 1:
                token_ids = token_ids[-(max_len - 1):]
            x = token_ids + [0] * (max_len - 1 - len(token_ids))
            self.data.append((x, target_id, reward))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, target, reward = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float),
        )


def rlhf_train(model, experiences, vocab_size, device, lr=5e-5, epochs=3):
    """
    Fine-tune con policy gradient simplificado.

    Loss = -reward * log(probabilidad del token)

    - reward positivo: AUMENTA la probabilidad (refuerza)
    - reward negativo: DISMINUYE la probabilidad (penaliza)
    """
    if not experiences:
        print("  Sin experiencias para entrenar.")
        return model

    dataset = RLHFDataset(experiences)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, target, reward in loader:
            x = x.to(device)
            target = target.to(device)
            reward = reward.to(device)

            logits = model(x)
            last_logits = logits[:, -1, :]

            log_probs = F.log_softmax(last_logits, dim=-1)
            action_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

            loss = -(reward * action_log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    return model
