import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model: int, head_count: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if d_model % head_count != 0:
            raise ValueError("`head_count` is not divisor of `d_model`")
        
        self.d_model = d_model
        self.head_count = head_count

        self.d_k = d_model // head_count
        self.d_k_sqrt = self.d_k ** 0.5

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(
        self,
        enc_q: torch.tensor,
        enc_k: torch.tensor,
        enc_v: torch.tensor,
        mask: torch.tensor = None,
    ) -> torch.tensor:
        """Forward hook impl."""
        q = self.W_q(enc_q).view(enc_q.size(0), enc_q.size(1), self.head_count, self.d_k).transpose(1, 2)  # [batch_size, seq_len, d_model] => [batch_size, seq_len, H, Hd] => [batch_size, H, seq_len, Hd]
        k = self.W_k(enc_k).view(enc_k.size(0), enc_k.size(1), self.head_count, self.d_k).transpose(1, 2)
        v = self.W_v(enc_v).view(enc_v.size(0), enc_v.size(1), self.head_count, self.d_k).transpose(1, 2)

        # A = softmax( (q * kT) / (sqrt(d_k)) ) * v
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k_sqrt)
        if mask is not None:
            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, v)                        # => [batch_size, H, seq_len, Hd]
        out = out.transpose(1, 2)                               # => [batch_size, seq_len, H, Hd]
        out = out.view(out.size(0), out.size(1), self.d_model)  # => [batch_size, seq_len, d_model]

        return self.W_o(out)
