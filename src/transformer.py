import torch
from torch import nn
from math import sqrt
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, embed_dim: int, *args, **kwargs,) -> None:
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        assert embed_dim % heads == 0, "embedding dimension should be divisible by the number of heads"

        self.heads = heads
        self.embed_dim = embed_dim
        self.W_q, self.W_k, self.W_v, self.W_o = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(4)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch,  head, length, d_tensor = q.shape
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q, k, v = self._split(q), self._split(k), self._split(v)

        scores = (q @ k.transpose(-2, -1)) / sqrt(d_tensor)

        scores = F.softmax(scores, dim=-1)
        v = scores @ v
        v = self._concat(v)
        out = self.W_o(v)
        return out

    def _split(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, length, embed_dim = input_tensor.shape
        head_dim = embed_dim // self.heads
        return input_tensor.reshape(batch, length, self.heads, head_dim).transpose(1, 2)

    def _concat(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, head, length, d_tensor = input_tensor.size()

        input_tensor = input_tensor.transpose(1, 2).contiguous().view(
            batch_size, length, self.embed_dim)
        return input_tensor
