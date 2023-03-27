import os
import numpy as np
import torch
import matplotlib.pyplot as plt



class TorqueTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, input_dim, path_to_txt, meanstd = {}, norm = True, is_train = True, visualize = False):
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
        x = self.dataset[idx, :self.input_len]
        y = self.dataset[idx, self.input_len:]

        if self.visualize:
            ax = plt.axes()
            ax.plot(x[:len(x)//2], 'r', label='qDot')
            ax.plot(x[len(x)//2:], 'g', label='tau')
            label = 'tau = {}'.format(y[0])
            ax.text(0.5, 0.5, label,
                    bbox=dict(facecolor='red', alpha=0.5),
                    transform=ax.transAxes)
            plt.legend()
            plt.show()

        if self.norm:
            sample = {'input': (x-self.mean)/self.std, 'label': y}
        else:
            sample = {'input': x, 'label': y}
        return sample['input'], sample['label']
