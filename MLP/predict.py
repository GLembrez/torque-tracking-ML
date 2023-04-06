import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from net import MLP
from dataloader import TorqueTrackingDataset

EVAL_BATCH_SIZE = 1


def initialize_net(trained_weights):
    obs_len = 5
    input_len = 2*obs_len
    net = MLP(input_dim=input_len, hidden_dim=128)
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

def get_predictions(args):
    #load weights and set seeed
    obs_len = 5
    input_len = 2*obs_len
    net = initialize_net(args.model)

    criterion = torch.nn.MSELoss().cuda()

    with torch.no_grad():
        losses = []
        with open(args.dataset, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        dataset = torch.from_numpy(np.array(lines)).float()
        T = dataset.shape[0]
        targets_list = np.zeros(T)
        output_list = np.zeros(T)
        alpha_list = np.zeros(T)

        for t in range(T-5):
            x = dataset[t:t+5, 1:3].reshape(10)
            y = dataset[t+5, 3].reshape(1)

            out = net(x)

            loss = criterion(out, y.cuda())
            losses.append(loss.data)
            output_list[t] = out.cpu().numpy()[0]
            targets_list[t] = y.cpu().numpy()[0]
            alpha_list[t] = x.cpu().numpy()[5]

    print(sum(losses)/len(losses))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot([0.001*t for t in range(T)], output_list, color = 'teal', label = 'prediction', linewidth=0.7)
    ax1.plot([0.001*t for t in range(T)], targets_list, color = 'lightsalmon', label = 'torque error', linewidth=0.7)
    ax1.set_xlim((0,2.5))
    plt.legend()
    # ax1.set_xlabel("t [s]")
    ax1.set_ylabel("torque [Nm]")
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot([0.001*t for t in range(T)], output_list-targets_list, color = 'teal', label = 'estimation error', linewidth=1)
    ax2.set_xlim((0,2.5))
    plt.legend()
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("torque [Nm]")
    plt.show()
    ax1.set_title("RKP")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    required=True,
                    type=str
    )
    ap.add_argument('--dataset',
                    required=True,
                    type=str
    )
    ap.add_argument('-v', '--verbose',
                    action='store_true'
    )
    ap.add_argument('--visualize',
                    action='store_true'
    )
    opt = ap.parse_args()

    print("Evaluating: ", opt.model)
    print("Dataset path: ", opt.dataset)

    #get predictions on dataset and evaluate wrt ground truth
    get_predictions(opt)
    return

if __name__=='__main__':
    main()
