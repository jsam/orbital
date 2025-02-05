import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, ctx_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pe = torch.zeros(ctx_len, d_model)

        # pe(pos, 2i) = sin( pos / ( 1e4 ** (i / d_model) ) )
        # pe(pos, 2i+1) = cos( pos / ( 1e4 ** (i / d_model) ))
        positions = torch.arange(start=0, end=ctx_len).float().unsqueeze(1)
        embeddings_indices = torch.arange(start=0, end=d_model, step=2).float()

        # div_term = 1 / (1e4 ** (i / d_model))
        div_term = 1 / (1e4 ** (embeddings_indices / d_model))

        self.pe[:, 0::2] = torch.sin(positions * div_term)
        self.pe[:, 1::2] = torch.cos(positions * div_term)

        # Register buffer for the GPU usage
        self.register_buffer("pe", self.pe)

    def forward(self, word_embeddings: torch.tensor) -> torch.tensor:
        return word_embeddings + self.pe[: word_embeddings.size(0), :]
