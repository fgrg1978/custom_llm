"""
RLHF with REINFORCE (inspired by Karpathy's nanochat).

Key improvements over basic policy gradient:
1. Multiple samples per position (generate N candidates, keep best)
2. Advantage normalization (reward - mean)
3. On-policy (fresh data each training step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class RLHFDataset(Dataset):
    """Dataset with advantages for REINFORCE."""

    def __init__(self, experiences, max_len=256):
        """
        experiences: list of (token_ids, target_id, advantage)
        """
        self.data = []
        for token_ids, target_id, advantage in experiences:
            if len(token_ids) >= max_len - 1:
                token_ids = token_ids[-(max_len - 1):]
            x = token_ids + [0] * (max_len - 1 - len(token_ids))
            self.data.append((x, target_id, advantage))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, target, advantage = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(advantage, dtype=torch.float),
        )


def normalize_rewards(experiences):
    """
    Convert raw rewards to advantages: advantage = reward - mean.
    Like Karpathy's nanochat: centers rewards so the model learns
    what's BETTER than average, not just what's good/bad.
    """
    if not experiences:
        return experiences

    rewards = [r for _, _, r in experiences]
    mean_reward = sum(rewards) / len(rewards)

    normalized = []
    for token_ids, target_id, reward in experiences:
        advantage = reward - mean_reward
        normalized.append((token_ids, target_id, advantage))

    return normalized


def rlhf_train(model, experiences, vocab_size, device, lr=5e-5, epochs=3):
    """
    REINFORCE with advantage normalization.

    Loss = -advantage * log(prob of chosen token)

    - advantage > 0: move was BETTER than average → increase probability
    - advantage < 0: move was WORSE than average → decrease probability
    - advantage = 0: average move → no change
    """
    if not experiences:
        print("  No experiences to train on.")
        return model

    # Normalize rewards to advantages
    experiences = normalize_rewards(experiences)

    dataset = RLHFDataset(experiences)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, target, advantage in loader:
            x = x.to(device)
            target = target.to(device)
            advantage = advantage.to(device)

            logits = model(x)
            last_logits = logits[:, -1, :]

            log_probs = F.log_softmax(last_logits, dim=-1)
            action_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

            # REINFORCE: -advantage * log_prob
            loss = -(advantage * action_log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    return model
