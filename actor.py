import torch
import torch.nn as nn
import torch.nn.functional as F

"""
input: state s
output: action a 
TODO: tune hidden layer number & size
"""
class Actor(nn.Module):

    def __init__(self, dim_state, hidden_size=[300,600]) -> None:
        super().__init__()
        hidden_size1, hidden_size2 = hidden_size

        self.hidden_layer = nn.Sequential(
            nn.Linear(dim_state, hidden_size1), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(inplace=True),
        )

        self.fc_steer = nn.Linear(hidden_size2, 1)
        self.fc_acc = nn.Linear(hidden_size2, 1)
        self.fc_brake = nn.Linear(hidden_size2, 1)

        # initialize layer weight
        nn.init.normal_(self.fc_steer.weight, 0, 1e-4)
        nn.init.normal_(self.fc_acc.weight, 0, 1e-4)
        nn.init.normal_(self.fc_brake.weight, 0, 1e-4)
    
    def forward(self, s):
        s = self.hidden_layer(s)
        
        out_steer = torch.tanh(self.fc_steer(s)) # [-1,1]
        out_acc = torch.sigmoid(self.fc_acc(s)) # [0,1]
        out_brake = torch.sigmoid(self.fc_brake(s)) # [0,1]

        return torch.cat((out_steer, out_acc, out_brake), dim=1)