import numpy as np
import torch



class TorqueTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_len, df):
        self.dataset = df                   # pandas dataframe
        self.sequence_len = sequence_len    # LSTM sequence length
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idt):
        """
        Returns the sequence  of inputs x[idt] -> x[idt+sequence_len]
        And the corresponding target (model(x)=y if model is perfect)

        q     - real position of the joints (sequence_len,7)
        alpha - real velocities of the joints (sequence_len,7)
        tau   - desired torque  (sequence_len,7)
        c     - gravity coriolis vector (sequence_len,7)
        x     - input for the model at time idt (sequence_len,28)
        y     - target for the model at time idt (7,)
        """
        time_len = self.__len__()
        if idt+self.sequence_len < time_len :
            # compute the input tensor
            q = np.asarray(self.dataset['q'][idt:idt+self.sequence_len].values.tolist())
            alpha = np.asarray(self.dataset['alpha'][idt:idt+self.sequence_len].values.tolist())
            tau = np.asarray(self.dataset['tau_d'][idt:idt+self.sequence_len].values.tolist())
            c = np.asarray(self.dataset['c'][idt:idt+self.sequence_len].values.tolist())
            y = self.dataset['f_point'][idt+self.sequence_len]
            x = np.concatenate((q,alpha,c,tau),axis=1)
        else :
            # if size of sequence exceeds limit of dataset start from begining
            return self.__getitem__(idt+self.sequence_len - time_len)
        return x,y
