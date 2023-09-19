import numpy as np
import torch
import torch.nn as nn
# trying to implement LSTM here


class my_lstm(nn.Module):
    def __init__(self, input_size, hidden_size,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden = torch.rand(hidden_size)
        self.cell_state = torch.rand(hidden_size)

        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid()
        )
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid()
        )
        self.cell_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        hx = torch.concat((x, self.hidden))
        self.cell_state *= self.forget_gate(hx)
        self.cell_state += self.cell_gate(hx)*self.input_gate(hx)
        out = torch.tanh(self.cell_state) * self.output_gate(hx)
        self.hidden = out
        return out
