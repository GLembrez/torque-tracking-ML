import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTM(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_features = num_features    #number of features (output)
        self.num_layers = num_layers        #number of layers
        self.input_size = input_size        #input size
        self.hidden_size = hidden_size      #hidden state
        self.seq_length = seq_length        #sequence length
        
        # LSTM layers :
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)  
        # fully connected layers :        
        self.fc_1 =  nn.Linear(hidden_size, 128)                            
        self.fc = nn.Linear(128, num_features)                             

        self.relu = nn.ReLU()
    
    def forward(self,x,h_n,c_n):
        # size of x : N_batch x input_size

         # Propagate input through LSTM
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n)) #lstm with input, hidden, and internal state
        out = h_n.view(-1, self.hidden_size)          #reshaping the data for Dense layer next
        out = self.relu(out)     #relu
        out = self.fc_1(out)    #first Dense
        out = self.relu(out)    #relu
        out = self.fc(out)      #Final Output
        return out, h_n, c_n