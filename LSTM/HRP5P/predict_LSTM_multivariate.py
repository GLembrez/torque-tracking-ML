import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from net_LSTM import LSTM
from dataloader_LSTM import TorqueTrackingDataset

EVAL_BATCH_SIZE = 1

joints=["RCY", "RCR", "RCP", "RKP", "RAP", "RAR",     # right leg
        "LCY", "LCR", "LCP", "LKP", "LAP", "LAR",     # left leg
        "WP","WR","WY",                               # waist
        "RSC","RSP","RSR","RSY","REP",                # right arm
        "LSC","LSP","LSR","LSY","LEP"]                # left arm

jointIdx = {"RCY" : 0, "RCR" : 1, "RCP" : 2, "RKP" : 3, "RAP" : 4, "RAR" : 5,     
        "LCY" : 6, "LCR" : 7, "LCP" : 8, "LKP" : 9, "LAP" : 10, "LAR" : 11,     
        "WP" : 12,"WR" : 13,"WY" : 14,                               
        "RSC" : 17,"RSP" : 18,"RSR" : 19,"RSY" : 20,"REP" : 21,                
        "LSC" : 35,"LSP" : 36,"LSR" : 37,"LSY" : 38,"LEP" : 39}   

def initialize_net(trained_weights):
    input_len = 50
    sequence_len = 10
    net = LSTM(num_features=25, input_size=input_len, hidden_size=100, num_layers=3, seq_length=sequence_len)
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
    input_len = 50
    sequence_len = 10
    net = initialize_net(args.model)


    criterion = torch.nn.MSELoss().cuda()

    with torch.no_grad():
        losses = []
        with open(args.dataset, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        dataset = torch.from_numpy(np.array(lines)).float()
        T = dataset.shape[0]
        targets_list = np.zeros((25,T-sequence_len-1))
        output_list = np.zeros((25,T-sequence_len-1))
        alpha_list = np.zeros((25,T-sequence_len-1))

        for t in range(T-sequence_len-1):
            x = dataset[t:t+sequence_len, 1:input_len+1]
            y = dataset[t:t+sequence_len, input_len+1:]

            out = net(x)

            loss = criterion(out, y.cuda())
            losses.append(loss.data)
            output_list[:,t] = out.cpu().numpy()[-1,:]
            targets_list[:,t] = y.cpu().numpy()[-1,:]
            alpha_list[:,t] = x.cpu().numpy()[-1,:25]

    print(sum(losses)/len(losses))
    
    fig = plt.figure()
    for i in [2,3,4] : 
        ax1 = fig.add_subplot(2,3,1+(i-2))
        ax1.plot([0.001*t for t in range(T-sequence_len-1)], output_list[i], color = 'teal', label = 'prediction', linewidth=0.7)
        ax1.plot([0.001*t for t in range(T-sequence_len-1)], targets_list[i], color = 'lightsalmon', label = 'torque error', linewidth=0.7)
        ax1.set_xlim((0,2.5))
        plt.legend()
        # ax1.set_xlabel("t [s]")
        ax1.set_ylabel("torque [Nm]")
        ax2 = fig.add_subplot(2,3,4+(i-2))
        ax2.plot([0.001*t for t in range(T-sequence_len-1)], output_list[i]-targets_list[i], color = 'teal', label = 'estimation error', linewidth=1)
        ax2.set_xlim((0,2.5))
        plt.legend()
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("torque [Nm]")
        ax1.set_title(joints[i])
    plt.show()


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
