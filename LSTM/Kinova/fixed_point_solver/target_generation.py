import os
import logging
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn

from net import LSTM


####################### HYPERPARAMETERS ###########################

n_DOFs = 7
input_len = 3
sequence_len = 30
num_layers = 2
hidden_size = 64
epsilon = 1e-3          # threshold on the improvement of the valid loss
dt = 5*1e-3             # simulation time step x sampling rate
W = torch.diag(torch.tensor([5,10,10,5,1,1,0.1]))

###################################################################


def initialize_net(trained_weights,
                   input_len,
                   sequence_len,
                   num_features,
                   hidden_size,
                   num_layers
                   ):
    
    net = LSTM(num_features=num_features,
               input_size=input_len*num_features, 
               hidden_size=hidden_size, 
               num_layers=num_layers, 
               seq_length=sequence_len)
    
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator',required=True,type=str)
    ap.add_argument("-o", "--out", required=True, default=False)
    ap.add_argument("-d", "--dataset", required=True)
    args = ap.parse_args()

    
    open(args.out, 'w').close()


    # Set manual seeds and defaults
    manual_seed = 0
    logging.info("Random Seed: %d", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type(torch.FloatTensor)


    estimator = initialize_net(args.estimator,
                               input_len,
                               sequence_len,
                               n_DOFs,
                               hidden_size,
                               num_layers)


    with torch.no_grad():
        with open(args.dataset, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        dataset = torch.from_numpy(np.array(lines)).float()
        T = dataset.shape[0]
        # x = dataset[:sequence_len, 1:input_len*n_DOFs+1].cuda()
        for t in tqdm(range(T-sequence_len-1)):
            # x[:sequence_len-1] = x.clone()[1:sequence_len]
            # x[sequence_len-1] = dataset[t,1:input_len*n_DOFs+1]
            x = dataset[t:t+sequence_len, 1:input_len*n_DOFs+1].cuda()
            x_numpy = x.detach().cpu().numpy() # use the raw inputs as inputs for the new model
            out = estimator(x)
            try :
                #   Newton Raphson loop
                n_iter = 0
                epsilon = 1e-3
                step = 0.1
                J = torch.zeros(n_DOFs,n_DOFs).cuda()
                x_init = x
                x_init[sequence_len-1,14:21] += out[sequence_len-1]
                F_eval = - out[sequence_len-1] + estimator(x_init)[sequence_len-1] 
                while F_eval.norm()>epsilon and n_iter < 10 :   
                    for i in range(n_DOFs) :
                        dx = torch.zeros(sequence_len,input_len*n_DOFs).cuda()
                        dx[sequence_len-1,14+i] = epsilon
                        out_new = estimator(x+dx)
                        J[:,i] = (out_new[sequence_len-1] - out[sequence_len-1]) / epsilon 
                        J[i,i] -= 1 
                    update = step * torch.matmul(J.inverse(), F_eval)
                    x[sequence_len-1,14:21] = x[sequence_len-1,14:21] - update
                    out_new = estimator(x)
                    F_eval = out[sequence_len-1] - out_new[sequence_len-1]
                    out[sequence_len-1] = out_new[sequence_len-1] 
                    n_iter += 1
                #     print(F_eval.norm())
                # print(n_iter - 1)
            except :
                pass


            
            out_numpy = out.detach().cpu().numpy()
            t      =    t
            q      =    x_numpy[sequence_len-1,:7]
            qDot   =    x_numpy[sequence_len-1,7:14]
            tauRef =    x_numpy[sequence_len-1,14:21]
            error  =    out_numpy[sequence_len-1]


            outline =       repr(t)      + \
                ',' + ','.join(map(str, q))   + \
                ',' + ','.join(map(str, qDot))   + \
                ',' + ','.join(map(str, tauRef)) + \
                ',' + ','.join(map(str, error)) 

            with open(args.out, 'a') as f:
              f.write(outline + '\n')

    return

if __name__=='__main__':
    main()