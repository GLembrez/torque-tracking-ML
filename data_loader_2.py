import os
import numpy as np
import torch
import pandas as pd



class TorqueTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.dataset = df
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idt):
        alpha_d = self.dataset['alpha_d'][idt]
        tau = self.dataset['tau_d'][idt]
        c = self.dataset['c'][idt]
        dtau = self.dataset['dtau'][idt]
        y = self.dataset['f_point'][idt]
        x = np.concatenate((alpha_d,c,tau,dtau),axis=0)
        return x,y
