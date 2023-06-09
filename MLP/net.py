import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
