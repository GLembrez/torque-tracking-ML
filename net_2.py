import torch
import torch.nn as nn
from torch.autograd import Variable 

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.model = nn.Sequential(nn.Linear(self.input_size,self.hidden_size),
                              nn.ReLU(),
                              nn.Linear(self.hidden_size,self.hidden_size),
                              nn.ReLU(),
                              nn.Linear(self.hidden_size,self.output_size))

    
    def forward(self,x):
        out = self.model(x)
        return out
