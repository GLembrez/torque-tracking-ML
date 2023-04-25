import random
import argparse
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


from net import LSTM
from data_loader import TorqueTrackingDataset

####################### HYPERPARAMETERS ###########################

n_DOFs = 7
input_len = 2
sequence_len = 20
num_layers = 3
hidden_size = 64
epsilon = 5*1e-3 # threshold on the improvement of the valid loss

###################################################################


def train(net, train_data, criterion, optimizer):
    net.train()
    losses = []

    for inputs, targets in train_data:
        inputs  = torch.autograd.Variable(inputs.cuda())
        targets = torch.autograd.Variable(targets.cuda())
        
        batch_size = inputs.size(0)
        out = net(inputs)
        loss = criterion(out, targets.view(-1, n_DOFs))



        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(losses)/len(losses)

@torch.no_grad()
def valid(net, valid_data, criterion):
    net.eval()
    error = []
    with torch.no_grad():
        for inputs, targets in valid_data:
            inputs = torch.autograd.Variable(inputs.cuda())
            targets = torch.autograd.Variable(targets.cuda())
            batch_size = inputs.size(0)
            out = net(inputs)
            error.append(criterion(out, targets.view(-1, n_DOFs)).data)
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
    h_n = torch.autograd.Variable(torch.zeros(num_layers, args.batch_size, hidden_size)) #hidden state
    c_n = torch.autograd.Variable(torch.zeros(num_layers, args.batch_size, hidden_size)) #internal state
        


    ncpus = os.cpu_count()
    # Set train and valid dataloaders
    train_set = TorqueTrackingDataset(input_len,n_DOFs,sequence_len, os.path.join(args.dataset, 'train.txt'), is_train=True)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
    logging.info("train data: %d batches of batch size %d", len(train_data), args.batch_size)

    meanstd = {'mean': train_set.mean, 'std':train_set.std}
    valid_set = TorqueTrackingDataset(input_len,n_DOFs, sequence_len, os.path.join(args.dataset, 'valid.txt'), meanstd, is_train=False)
    valid_data = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
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

        # if epoch>0 and abs(log_valid_loss[epoch]-log_valid_loss[epoch-1])<epsilon:
        #     torch.save(net.state_dict(), os.path.join(args.outdir, "trained.model"))
        #     torch.save(net, os.path.join(args.outdir, "trained.pth"))

        #     fig = plt.figure()
        #     plt.plot(log_train_loss,color='teal',linewidth=2,label='Train Loss')
        #     plt.plot(log_valid_loss,color='lightsalmon',linewidth=2,label='Valid Loss')
        #     plt.legend()
        #     plt.xlabel('epoch')
        #     plt.ylabel('MSE loss')
        #     plt.title('Model training performances')
        #     plt.show()

        #     print(epoch)
        #     return

    torch.save(net.state_dict(), os.path.join(args.outdir, "trained.model"))
    torch.save(net, os.path.join(args.outdir, "trained.pth"))

    fig = plt.figure()
    plt.plot(log_train_loss,color='teal',linewidth=2,label='Train Loss')
    plt.plot(log_valid_loss,color='lightsalmon',linewidth=2,label='Valid Loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('Model training performances')
    plt.show()

if __name__=='__main__':
    main()

