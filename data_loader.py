import os
import numpy as np
import torch
import pandas as pd



class TorqueTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, input_dim, n_DOFs, sequence_len, df):

        self.dataset = df

        # hardcoded
        self.input_len = input_dim
        self.sequence_len = sequence_len
        self.n_DOFs = n_DOFs

        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idt):
        time_len = self.__len__()
        if idt+self.sequence_len < time_len :
            q = np.asarray(self.dataset['q'][idt:idt+self.sequence_len].values.tolist())
            alpha = np.asarray(self.dataset['alpha'][idt:idt+self.sequence_len].values.tolist())
            tau = np.asarray(self.dataset['tau_d'][idt:idt+self.sequence_len].values.tolist())
            c = np.asarray(self.dataset['c'][idt:idt+self.sequence_len].values.tolist())
            y = self.dataset['tau_f'][idt+self.sequence_len]
            x = np.concatenate((q,alpha,c,tau),axis=1)
        else :
            return self.__getitem__(idt+self.sequence_len - time_len)
        return x,y
