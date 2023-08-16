import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTM(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_features = num_features    # number of features (output)
        self.num_layers = num_layers        # number of layers
        self.input_size = input_size        # input size
        self.hidden_size = hidden_size      # hidden state
        self.seq_length = seq_length        # sequence length
        
        # LSTM layers :
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)  
        # fully connected layers :                          
        self.fc = nn.Linear(hidden_size, num_features)                                
        self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()
    
    def forward(self,x):
        # size of x : N_batch x input_size
        if len(x.shape) == 3 :
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) #internal state
        else :
            h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size).cuda()) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size).cuda()) #internal state
        
         # Propagate input through LSTM
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))    
        if len(x.shape) == 3 :   
            out = self.dropout(output)[:,-1,:]
        else:
            out = self.dropout(output)[-1,:]
        out = self.fc(out)                              #Final Output
        return out
    
    # def forward(self,x,h,c):
    #     output, (h_n, c_n) = self.lstm(x, (h, c))
    #     out = self.dropout(output)[-1,:]
    #     out = self.fc(out)
    #     return out,h_n,c_n
