import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from net import LSTM

EVAL_BATCH_SIZE = 1



def initialize_net(trained_weights):
    input_len = 14
    sequence_len = 10
    net = LSTM(num_features=7, input_size=input_len, hidden_size=64, num_layers=3, seq_length=sequence_len)
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
    input_len = 14
    sequence_len = 10
    net = initialize_net(args.model)
    dt = 5*1e-3


    criterion = torch.nn.MSELoss(reduction='mean').cuda()

    with torch.no_grad():
        losses = []
        with open(args.dataset, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        dataset = torch.from_numpy(np.array(lines)).float()
        T = dataset.shape[0]
        targets_list = np.zeros((7,T-sequence_len-1))
        output_list = np.zeros((7,T-sequence_len-1))
        alpha_list = np.zeros((7,T-sequence_len-1))



        alpha_p = dataset[sequence_len, 1:8].cuda()

        for t in range(T-sequence_len-1):
            x = dataset[t:t+sequence_len, 1:input_len+1].cuda()
            y = dataset[t:t+sequence_len, input_len+1:22].cuda()
            C = dataset[t:t+sequence_len, 7*(2+1)+1:7*(2+2)+1].cuda()
            M = torch.reshape(dataset[t:t+sequence_len, 7*(2+2)+1:], (sequence_len,7, 7)).cuda()
            M_inv = torch.inverse(M[-1])

            alpha_r = x[-1,:7]
            out = net(x)
            if t>0:
                alpha_p[:,None] = alpha_p[:,None] + dt * torch.matmul(M_inv,(x[-1,7:,None]-y[-1,:,None]-C[-1,:,None]))
            dv = alpha_r-alpha_p
            

            
            loss = 0.92* torch.mean(torch.matmul(dv[None,:],torch.matmul(M.cuda()[:-1,:,:],dv[:,None]))) +  0.08*criterion(out, y.cuda())
            losses.append(loss.data)
            output_list[:,t] = out[-1,:].cpu().numpy()
            targets_list[:,t] = y[-1,:].cpu().numpy()
            alpha_list[:,t] = x.cpu().numpy()[-1,:7]

    print(sum(losses)/len(losses))
    
    fig = plt.figure()
    for i in range(7) : 
        ax1 = fig.add_subplot(7,1,1+i)
        ax1.set_title("joint {}".format(i+1))
        ax1.plot([0.001*t for t in range(T-sequence_len-1)], output_list[i], color = 'teal', label = 'prediction', linewidth=0.7)
        ax1.plot([0.001*t for t in range(T-sequence_len-1)], targets_list[i], color = 'lightsalmon', label = 'target', linewidth=0.7)
        plt.legend()

    
    # for i in range(7) : 
    #     fig = plt.figure()
    #     plt.title("joint {}".format(i+1))
    #     plt.plot(alpha_list[i], targets_list[i],'.', markersize=0.3, color = 'teal', label = 'torque error', linewidth=0.7)
    #     plt.plot(alpha_list[i], output_list[i],'.', markersize=0.3, color = 'lightsalmon', label = 'prediction', linewidth=0.7)
    #     plt.legend()
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
