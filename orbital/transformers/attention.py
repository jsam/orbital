import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(
        self,
        enc_q: torch.tensor,
        enc_k: torch.tensor,
        enc_v: torch.tensor,
        mask: torch.tensor = None,
    ) -> torch.tensor:
        """Forward hook impl."""
        q = self.W_q(enc_q)  # [batch_size, ctx_len, d_model]
        k = self.W_k(enc_k)
        v = self.W_v(enc_v)

        # A = softmax( (q * kT) / (sqrt(d_k)) ) * v
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(1) ** 0.5)
        if mask is not None:
            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        # [batch_size, ctx_len, d_model]
        return torch.matmul(attention, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, head_count: int, *args, **kwargs):
        super().__(*args, **kwargs)
        if d_model % head_count != 0:
            raise ValueError("d_model is not divisible by head_count")

        self.d_model = d_model
        self.head_count = head_count
        self.d_k = d_model // head_count  # Dimension of each head
