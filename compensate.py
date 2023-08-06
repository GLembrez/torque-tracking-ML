import pandas as pd
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
from progress.bar import Bar

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from net import LSTM
from data_loader import TorqueTrackingDataset
from simulation import Simulation


####################### HYPERPARAMETERS ###########################

n_DOFs = 7
input_len = 4
sequence_len = 10
num_layers = 2
hidden_size = 64
tol = 1e-2                  # threshold on the improvement of the valid loss
xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/xml/gen3_7dof_mujoco.xml"

###################################################################

def initialize_net(trained_weights):
    input_len = 28
    sequence_len = 10
    net = LSTM(num_features=7, input_size=input_len, hidden_size=64, num_layers=2, seq_length=sequence_len)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    net.load_state_dict(torch.load(trained_weights))
    manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    cudnn.deterministic=True
    cudnn.benchmark = False
    print("====Loaded weights====")
    return net

def clean_plot(df,loss):

    print(np.average(loss))
    loss_fig = plt.figure()
    plt.plot(loss,color = 'teal', label='compensation error')
    plt.xlabel("epoch")
    plt.ylabel("RMSE [Nm]")

    time_perf = plt.figure()
    for i in range(7):
        ax = time_perf.add_subplot(7,1,i+1)
        meas = [f[i] for f in df["f_point"]]
        pred = [t[i] for t in df["out"]]
        error = [t[i] for t in df["tau_f"]]
        ax.plot(meas, color = "teal", label = "measurement")
        ax.plot(pred,color='red',label="prediction")
        ax.plot(error,color='lightsalmon',label="real error")
        plt.legend()
        ax.set_ylabel("DOF "+str(i+1))
        ax.set_ylim([-50,50])
    ax.set_xlabel("iteration")

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    required=True,
                    type=str
    )
    args = ap.parse_args()

    net = initialize_net(args.model)
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    sim = Simulation(xml_path,visual=True)
    df,loss = sim.run_model(net)
    clean_plot(df,loss)

    return



if __name__=='__main__':
    main()