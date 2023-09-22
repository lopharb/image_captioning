import torch
import torch.nn as nn

# TODO add stuff like importance factor, attention (need to read the paper first), etc


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
