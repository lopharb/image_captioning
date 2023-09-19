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
        """
        Each tick is processed as such:
        1. Cell receives new input 
        2. The input is being concatted w the hidden state of the cell and stored somewhere as *hx*
        3. *hx* is passed through a FC layer and sigmoid (the forget gate)
        4. Cell state is multiplied by the result
        5. *hx* then goes through another FC layer activated with *sigmoid* 
        6. The result is multiplied by *hx* passed through a FC layer with *tanh*
        7. It then gets added to the cell state
        8. *hx* is passed through yet another FC layer with *sigmoid*
        9. New cell state is passed through *tanh* and added to the result of st. 8, which in the end forms the output value
        """
        hx = torch.concat((x, self.hidden))
        self.cell_state *= self.forget_gate(hx)
        self.cell_state += self.cell_gate(hx)*self.input_gate(hx)
        out = torch.tanh(self.cell_state) * self.output_gate(hx)
        self.hidden = out
        return out


smth = my_lstm(5, 50)
smth_else = nn.LSTM(input_size=5, hidden_size=50)
x = torch.rand(5)
y = smth(x)
x = torch.rand([1, 5])
print(nn.MSELoss()(smth_else(x)[0][0], y))
