#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size*2, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, encoder_outputs, mask):
        _, seq_length, _ = encoder_outputs.shape
        mask = mask[:, :seq_length].eq(0)
        energy = self.attention(encoder_outputs).squeeze()
        energy = energy.masked_fill(mask=mask, value=-float('inf'))
        weights = F.softmax(energy, dim=-1)  # (B, L)
        # (B, L, H) * (B, L, 1) -> (B, L)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(1)

        return outputs

