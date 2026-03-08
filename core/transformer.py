"""
Transformer decoder generico para prediccion de secuencias.
Arquitectura tipo GPT, reutilizable para cualquier dominio.
"""

import math
import torch
import torch.nn as nn


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_len=256,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x: (batch, seq_len) - secuencia de token IDs
        returns: (batch, seq_len, vocab_size) - logits
        """
        B, T = x.shape
        device = x.device

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions)
        h = self.drop(h)

        # Mascara causal (no puede ver el futuro)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Padding mask
        pad_mask = (x == 0)

        # Decoder con memory vacio (autoregresivo puro)
        memory = torch.zeros(B, 1, self.d_model, device=device)
        h = self.transformer(
            h,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )

        logits = self.out_proj(h)
        return logits
