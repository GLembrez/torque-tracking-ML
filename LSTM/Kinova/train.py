import random
import argparse
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from progress.bar import Bar

from net import LSTM
from data_loader import TorqueTrackingDataset

####################### HYPERPARAMETERS ###########################

n_DOFs = 7
input_len = 2
sequence_len = 10
num_layers = 3
hidden_size = 64
epsilon = 5*1e-3        # threshold on the improvement of the valid loss
dt = 5*1e-3             # simulation time step x sampling rate
regularization = 0.5     # loss regularization parameter

###################################################################


def train(net, train_data, criterion, optimizer):
    net.train()
    losses = []
    tbar = Bar('Training', max=len(train_data))
    for inputs, targets, C, M in train_data:

        batch_size = inputs.size(dim=0)
        M_inv = torch.inverse(M.cuda())
        C = C.cuda()
        inputs  = torch.autograd.Variable(inputs.cuda())
        targets = torch.autograd.Variable(targets.cuda())
        alpha_r = torch.diff(inputs[:,:,:n_DOFs], n=1, dim=1)
        alpha_p = torch.zeros(batch_size,sequence_len-1,n_DOFs).cuda()
        out = net(inputs)
        friction_estimation = out.reshape(batch_size, sequence_len, n_DOFs)
        for t in range(sequence_len-1):
            alpha_p[:,t,:,None] = dt * torch.matmul(M_inv[:,t,:,:],(inputs[:,t,n_DOFs:,None]-friction_estimation[:,t,:,None]-C[:,t,:,None]))
        dv = alpha_r-alpha_p
        loss =  (1-regularization) * criterion(out, targets.view(-1, n_DOFs)) + regularization * torch.mean(torch.matmul(dv[:,:,None,:],torch.matmul(M.cuda()[:,:-1,:,:],dv[:,:,:,None]))) 

        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.next()
    return sum(losses)/len(losses)

@torch.no_grad()
def valid(net, valid_data, criterion):
    net.eval()
    error = []
    ebar = Bar('Evaluating', max=len(valid_data))
    with torch.no_grad() :
        for inputs, targets, C, M in valid_data:

            batch_size = inputs.size(dim=0)
            M_inv = torch.inverse(M.cuda())
            C = C.cuda()
            inputs  = torch.autograd.Variable(inputs.cuda())
            targets = torch.autograd.Variable(targets.cuda())
            alpha_r = torch.diff(inputs[:,:,:n_DOFs], n=1, dim=1)
            alpha_p = torch.zeros(batch_size,sequence_len-1,n_DOFs).cuda()
            out = net(inputs)
            friction_estimation = out.reshape(batch_size, sequence_len, n_DOFs)
            for t in range(sequence_len-1):
                alpha_p[:,t,:,None] = dt * torch.matmul(M_inv[:,t,:,:],(inputs[:,t,n_DOFs:,None]-friction_estimation[:,t,:,None]-C[:,t,:,None]))
            dv = alpha_r-alpha_p
            #### replace sum with mean ?
            loss = (1-regularization)* criterion(out, targets.view(-1, n_DOFs)) + regularization * torch.mean(torch.matmul(dv[:,:,None,:],torch.matmul(M.cuda()[:,:-1,:,:],dv[:,:,:,None]))) 
            ebar.next()
            error.append(loss.data)
    return sum(error)/len(error)

def main():
    log_train_loss = []
    log_valid_loss = []

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", required=True, default=False)
    ap.add_argument("-e", "--epochs", required=False, default=100, type=int)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-c", "--checkpoint", required=False, default=50, type=int)
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

    # Set network model, loss criterion and optimizer
    net = LSTM(num_features=n_DOFs, input_size=input_len*n_DOFs, hidden_size=hidden_size, num_layers=num_layers, seq_length=sequence_len)
    net = torch.nn.DataParallel(net).cuda()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.rate)
    logging.info(repr(optimizer))
        


    ncpus = os.cpu_count()
    # Set train and valid dataloaders
    train_set = TorqueTrackingDataset(input_len,n_DOFs,sequence_len, os.path.join(args.dataset, 'train.txt'), is_train=True)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=ncpus, drop_last=True)
    logging.info("train data: %d batches of batch size %d", len(train_data), args.batch_size)

    meanstd = {'mean': train_set.mean, 'std':train_set.std}
    valid_set = TorqueTrackingDataset(input_len,n_DOFs, sequence_len, os.path.join(args.dataset, 'valid.txt'), meanstd, is_train=False)
    valid_data = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=ncpus, drop_last=True)
    logging.info("valid data: %d batches of batch size %d", len(valid_data), args.batch_size)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


    for epoch in range(args.epochs):
        if args.outdir and epoch%args.checkpoint==0:
            torch.save(net.state_dict(), os.path.join(args.outdir, "trained_" + str(int(epoch)) + ".model"))
        train_loss = train(net, train_data, criterion, optimizer)
        valid_loss = valid(net, valid_data, criterion)
        log_train_loss.append(train_loss.item())
        log_valid_loss.append(valid_loss.item())
        logging.info("iters: %d train_loss: %f valid_loss: %f lr: %f",
                     epoch,
                     train_loss.item(),
                     valid_loss.item(),
                     optimizer.param_groups[0]['lr'])
        if epoch%10==0:
            lr_scheduler.step()
        if epoch>0 and (abs(log_valid_loss[epoch]-log_valid_loss[epoch-1])<epsilon or epoch==args.epochs-1):
            torch.save(net.state_dict(), os.path.join(args.outdir, "trained.model"))
            torch.save(net, os.path.join(args.outdir, "trained.pth"))

            fig = plt.figure()
            plt.plot(log_train_loss,color='teal',linewidth=2,label='Train Loss')
            plt.plot(log_valid_loss,color='lightsalmon',linewidth=2,label='Valid Loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.title('Model training performances')
            plt.savefig('training_overview.png')

            print(epoch)
            return

    

if __name__=='__main__':
    main()

