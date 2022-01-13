import torch
import torch.nn as nn
import torch.nn.functional as F

"""
input: s, a
output: Q(s,a)
TODO: tune hidden layer number & size
"""
class Critic2(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_size=[256,512]) -> None:
        super().__init__()
        hidden_size1, hidden_size2 = hidden_size

        self.hidden_layer_s = nn.Sequential(
            nn.Linear(dim_state, hidden_size1),
            # nn.BatchNorm1d(hidden_size1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size1, hidden_size2),
        )

        self.hidden_layer_a = nn.Sequential(
            nn.Linear(dim_action, hidden_size2),
        )

        self.hidden_layer_out = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            # nn.BatchNorm1d(hidden_size2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size2, 1)
        )

    def forward(self, s, a):
        s = self.hidden_layer_s(s)
        a = self.hidden_layer_a(a)

        output = self.hidden_layer_out(s+a) # B*1

        return output


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_size=[300,600,600]) -> None:
        super().__init__()
        hidden_size1, hidden_size2, hidden_size3 = hidden_size

        self.fc1 = nn.Sequential(
            nn.Linear(dim_state, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(dim_action, hidden_size2),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(hidden_size3, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(inplace=True),
        )

        self.fc5 = nn.Linear(hidden_size3, 1)

    def forward(self, s, a):
        s = self.fc1(s)
        s = self.fc2(s)
        a = self.fc3(a)
        x = self.fc4(s + a)

        output = self.fc5(x) # B*1

        return output

