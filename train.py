import os
import logging
import random
import argparse
import torch
from progress.bar import Bar
from torch.utils.data import DataLoader
from net import LSTM
from data_loader import TorqueTrackingDataset
from simulation import Simulation
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np

####################### HYPERPARAMETERS ###########################

n_DOFs = 7
input_len = 4
sequence_len = 10
num_layers = 2
hidden_size = 32
tol = 1e-2              # threshold on the improvement of the valid loss
T_train = 3600*1000     # Number of training steps
T_valid = 600*1000      # Number of validation steps
xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/xml/gen3_7dof_mujoco.xml"

###################################################################


def plot_loss(valid,train):
    """
    plots validation and training loss
    """
    fig = plt.figure()
    plt.plot(valid, color='teal', label='valid loss')
    plt.plot(train, color='lightsalmon', label='train loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.show()
    


def train(net, train_data, criterion, optimizer):
    """
    trains the model net through the whole dataset train_data once
    returns the average loss during the epoch

    inputs  - tensor of shape (n_batch, sequence_len, n_input)
    targets - tensor of shape (n_batch, n_input)
    out     - tensor of shape (n_batch, n_input)
    """
    net.train()
    losses = []
    tbar = Bar('Training', max=len(train_data))
    for inputs, targets in train_data:
        inputs  = torch.autograd.Variable(inputs.float().cuda())
        targets = torch.autograd.Variable(targets.float().cuda())
        out = net(inputs)
        loss = criterion(out, targets)  
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        tbar.next()

    return sum(losses)/len(losses)


@torch.no_grad()
def valid(net, valid_data, criterion):
    """
    evaluates net on the valid dataset
    returns the average loss 
    gradients are not computed to speed the process

    inputs  - tensor of shape (n_batch, sequence_len, n_input)
    targets - tensor of shape (n_batch, n_input)
    out     - tensor of shape (n_batch, n_input)
    """
    net.eval()
    error = []
    with torch.no_grad():
        for inputs, targets in valid_data:
            inputs = torch.autograd.Variable(inputs.float().cuda())
            targets = torch.autograd.Variable(targets.float().cuda())
            out = net(inputs)
            error.append(criterion(out, targets))
    return sum(error)/len(error)



def initialize_net(trained_weights,
                   input_len,
                   sequence_len,
                   num_features,
                   hidden_size,
                   num_layers
                   ):
    """
    Uses stored weights to initialize a pre-trained network and returns it
    """
    
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
    ap.add_argument("-c", "--checkpoint", required=False, default=50, type=int)
    ap.add_argument("-m", "--model", required=False, default=None)
    ap.add_argument("--batch_size", required=False, default=8, type=int)
    ap.add_argument("--rate", required=False, default=1e-3, type=float)
    args = ap.parse_args()

    # create mujoco simulations
    sim1 = Simulation(xml_path)
    sim2 = Simulation(xml_path)
    # generate training and validation datasets
    df_train = sim1.simulate(T_train)
    df_valid = sim2.simulate(T_valid)


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


    # create model and set to parallel
    net = LSTM(num_features=n_DOFs,
               input_size=input_len*n_DOFs, 
               hidden_size=hidden_size, 
               num_layers=num_layers, 
               seq_length=sequence_len)
    net = torch.nn.DataParallel(net).cuda()

    # initialize training
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    logging.info(repr(optimizer))
    ncpus = os.cpu_count()

    # Creates training and validation data loaders
    train_set = TorqueTrackingDataset(sequence_len, df_train)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
    logging.info("train data: %d batches of batch size %d", len(train_data), args.batch_size)
    valid_set = TorqueTrackingDataset(sequence_len, df_valid)
    valid_data = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
    logging.info("valid data: %d batches of batch size %d", len(valid_data), args.batch_size)

    # train loop
    with torch.autograd.set_detect_anomaly(True):
        valid_loss = valid(net, valid_data, criterion)
        log_valid = []
        log_train = []
        try:
            for epoch in range(args.epochs):
                if args.outdir and epoch%args.checkpoint==0:
                    # save every checkpoint epoch
                    torch.save(net.state_dict(), os.path.join(args.outdir, "trained_" + str(int(epoch)) + ".model"))
                
                # train and evaluate the model
                train_loss = train(net, train_data, criterion, optimizer)
                valid_loss_new = valid(net, valid_data, criterion)

                if torch.abs(valid_loss_new-valid_loss)<tol:
                    # Save if threshold on improvement over the valid data is met
                    torch.save(net.state_dict(), os.path.join(args.outdir, "trained.model"))
                    plot_loss(log_valid,log_train)
                    break
                
                # update losses
                valid_loss = valid_loss_new
                log_valid.append(valid_loss.cpu().numpy())
                log_train.append(train_loss.cpu().numpy())
                logging.info(" iters: %d valid_loss: %f" ,
                            epoch,
                            valid_loss.item())
                if epoch%10==0:
                    lr_scheduler.step()
            # if finishes without metting stagnation
            torch.save(net.state_dict(), os.path.join(args.outdir, "trained.model"))
            plot_loss(log_valid,log_train)
        except:
            # In case of interruption save anyway
            torch.save(net.state_dict(), os.path.join(args.outdir, "trained.model"))
            plot_loss(log_valid,log_train)

    return

if __name__=='__main__':
    main()
