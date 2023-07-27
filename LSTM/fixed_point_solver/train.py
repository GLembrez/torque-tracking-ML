import os
import logging
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


def train(net, train_data, criterion, optimizer):

    net.train()
    losses = []
    tbar = Bar('Training', max=len(train_data))
    for inputs, targets in train_data:
        inputs  = torch.autograd.Variable(inputs.cuda())
        targets = torch.autograd.Variable(targets.cuda())
        out = net(inputs)
        loss = criterion(out, targets.view(-1, n_DOFs))  
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        tbar.next()

    return sum(losses)/len(losses)







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
    ap.add_argument("-o", "--outdir", required=True, default=False)
    ap.add_argument("-e", "--epochs", required=False, default=100, type=int)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-c", "--checkpoint", required=False, default=50, type=int)
    ap.add_argument("-m", "--model", required=False, default=None)
    ap.add_argument("--batch_size", required=False, default=8, type=int)
    ap.add_argument("--rate", required=False, default=1e-4, type=float)
    args = ap.parse_args()

    # Create output directory if necessary
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Set up loggers
    log_file_path = os.path.join(args.outdir, 'train.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        filename=log_file_path,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)

    logging.info("Training for a total of %d epochs", args.epochs)
    logging.info("Weights are saved to %s after every %d epochs", args.outdir, args.checkpoint)
    logging.info("Dataset path: %s", args.dataset)

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


    # net = initialize_net(args.estimator,
    #                            input_len,
    #                            sequence_len,
    #                            n_DOFs,
    #                            hidden_size,
    #                            num_layers)

    net = LSTM(num_features=n_DOFs,
               input_size=input_len*n_DOFs, 
               hidden_size=hidden_size, 
               num_layers=num_layers, 
               seq_length=sequence_len)
    
    net = torch.nn.DataParallel(net).cuda()

    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    logging.info(repr(optimizer))
    ncpus = os.cpu_count()

    train_set = TorqueTrackingDataset(input_len,n_DOFs,sequence_len, os.path.join(args.dataset, 'train.txt'))
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
    logging.info("train data: %d batches of batch size %d", len(train_data), args.batch_size)

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.epochs):
            if args.outdir and epoch%args.checkpoint==0:
                torch.save(net.state_dict(), os.path.join(args.outdir, "trained_" + str(int(epoch)) + ".model"))
            train_loss = train(net, train_data, criterion, optimizer)
            logging.info(" iters: %d train_loss: %f" ,
                        epoch,
                        train_loss.item())
            if epoch%10==0:
                lr_scheduler.step()


    return

if __name__=='__main__':
    main()
