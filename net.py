import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        """
        Structure of the model

        num_layers of LSTM   - input_size x hidden_size  
        Dropout              - probability of discarding each weight on the input tensor 
        Fully-connected      - hidden_sie x output_size
        """

        self.output_size = output_size    # number of features (output)
        self.num_layers = num_layers        # number of layers
        self.input_size = input_size        # input size
        self.hidden_size = hidden_size      # hidden state
        self.seq_length = seq_length        # sequence length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)  
        # fully connected layer                      
        self.fc = nn.Linear(hidden_size, output_size)     
        # dropout layer                           
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        """
        Propagates x through the network

        x   - tensor of shape N_batch x sequence_size x input_size
        h_0 - initial state of the hidden layer
        c_0 - initial state of the memory cell
        """
        if len(x.shape) == 3 :
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) #memory state
        else :
            h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size).cuda()) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size).cuda()) #memory state
        
        # Propagate input through LSTM
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))  
        # Apply dropout  
        if len(x.shape) == 3 :   
            out = self.dropout(output)[:,-1,:]
        else:
            out = self.dropout(output)[-1,:]
        # Resize through fully connected layer
        out = self.fc(out)                              
        return out

