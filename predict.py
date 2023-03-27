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

    #load dataset using dataloader
    meanstd = torch.load(os.path.join(os.path.dirname(args.dataset), 'mean.pth.tar'))
    eval_set = TorqueTrackingDataset(input_len,args.dataset, meanstd, is_train=False)
    eval_data = DataLoader(eval_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=1)
    print("valid data size is: {} batches of batch size: {}".format(len(eval_data), EVAL_BATCH_SIZE))



    with torch.no_grad():
        losses = []
        targs = []
        outs = []
        with open(args.dataset, 'r') as f:
            lines = [list(map(float, l.strip().split(','))) for l in f.readlines()]
        fullDataset = torch.from_numpy(np.array(lines)).float()
        cmdTorque = fullDataset[:,input_len-1]

        for batch_id, (inputs, targets) in enumerate(eval_data):
            #forward pass through network
            inp = torch.autograd.Variable(inputs.cuda())
            targets = torch.autograd.Variable(targets.cuda())
            out = net(inp)
            loss = criterion(out, targets)
            losses.append(loss.item())
            
            targs.append(targets.cpu().numpy()[0])
            outs.append(out.cpu().numpy()[0])


            #get evaluations
            if args.verbose:
                print("Batch: ", batch_id)
                print("Loss = ", loss.item())
                y = out.cpu().numpy()[0]
                y_ = targets.cpu().numpy()[0]

        #visualize iff necessary
        if args.visualize:
            fig = plt.figure()
            plt.plot(targs, color='teal', label = 'Torque error', linewidth=1)
            plt.plot(outs, color='lightsalmon', label='Predicted torque error',linewidth = 1)
            # plt.plot(cmdTorque,color='red',label='Command torque',linewidth=1)
            plt.legend()
            plt.show()
                            
        print("Mean loss: ", sum(losses)/len(losses))
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
