import os
import numpy as np
import torch
import matplotlib.pyplot as plt



class TorqueTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, input_dim, n_DOFs, sequence_len, path_to_txt):

        with open(path_to_txt, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        self.dataset = torch.from_numpy(np.array(lines)).float()

        # hardcoded
        self.input_len = input_dim
        self.sequence_len = sequence_len
        self.n_DOFs = n_DOFs

        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idt):
        time_len = self.dataset.size(0)
        if idt+self.sequence_len < time_len :
            x = self.dataset[idt:idt+self.sequence_len, 1:self.n_DOFs*self.input_len+1]
            y = self.dataset[idt:idt+self.sequence_len, self.n_DOFs*self.input_len+1:self.n_DOFs*(self.input_len+1)+1]
        else :
            return self.__getitem__(idt+self.sequence_len - time_len)
        return x,y
