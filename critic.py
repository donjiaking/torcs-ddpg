import torch
import torch.nn as nn
import torch.nn.functional as F

"""
input: s, a
output: Q(s,a)
TODO: tune hidden layer number & size
"""
class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_size=[100,200]) -> None:
        super().__init__()
        hidden_size1, hidden_size2 = hidden_size

        self.hidden_layer_s = nn.Sequential(
            nn.Linear(dim_state, hidden_size1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size1, hidden_size2),
        )

        self.hidden_layer_a = nn.Sequential(
            nn.Linear(dim_action, hidden_size2),
        )

        self.hidden_layer_out = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size2, 1)
        )

    def forward(self, s, a):
        s = self.hidden_layer_s(s)
        a = self.hidden_layer_a(a)

        output = self.hidden_layer_out(s+a)

        return output
