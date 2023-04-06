import os
import numpy as np
import torch
import matplotlib.pyplot as plt



class TorqueTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, input_dim, path_to_txt, meanstd = {}, norm = False, is_train = True, visualize = False):
        self.norm = norm
        self.is_train = is_train
        self.visualize = visualize

        with open(path_to_txt, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        self.dataset = torch.from_numpy(np.array(lines)).float()

        # hardcoded
        self.input_len = input_dim

        if meanstd == {}:
            # compute mean and std-dev
            self.mean = self.dataset[:, :self.input_len].mean(axis=0)
            self.std = self.dataset[:, :self.input_len].std(axis=0)
            meanstd = {'mean': self.mean,
                       'std': self.std}
        else:
            self.mean = meanstd['mean']
            self.std = meanstd['std']

        if self.is_train:
            data_dir = os.path.dirname(path_to_txt)
            torch.save(meanstd, os.path.join(data_dir, 'mean.pth.tar'))
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        time_len = self.dataset.size(0)
        if idx+5 < time_len :
            x = self.dataset[idx:idx+5, 1:3].reshape(10)
            y = self.dataset[idx+5, 3].reshape(1)
        else :
            x = torch.from_numpy(np.zeros(10)).float()
            y = torch.from_numpy(np.zeros(1)).float()

        if self.norm:
            sample = {'input': (x-self.mean)/self.std, 'label': y}
        else:
            sample = {'input': x, 'label': y}
        return sample['input'], sample['label']
