import random
import argparse
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


from net_GAN import LSTM
from data_loader_GAN import TorqueTrackingDataset

####################### HYPERPARAMETERS ###########################

n_DOFs = 7
sequence_len = 10
num_layers = 3
hidden_size = 64

###################################################################


def train_G(G,D, train_data,criterion, optimizer):
    G.train()
    losses = []

    for inputs in train_data:
        inputs  = torch.autograd.Variable(inputs.cuda())

        out = G(inputs)
        prediction = D(torch.add(out, inputs[:,:,:21]))[:,-1,0]
        loss = criterion(prediction,torch.ones(hidden_size).cuda())
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(losses)/len(losses)


def train_D(D,G, real_data,simu_data, criterion, optimizer):
    D.train()
    losses = []

    for inputs in simu_data:
        inputs  = torch.autograd.Variable(inputs.cuda())
        out = G(inputs)
        prediction = D(torch.add(out, inputs[:,:,:21]))[:,-1,0]
        loss = criterion(prediction,torch.ones(hidden_size).cuda())
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for sample in real_data:

        prediction = D(sample[:,:,:21])[:,-1,0]
        loss = criterion(prediction,torch.zeros(hidden_size).cuda())
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses)/len(losses)



def main():
    log_G_loss = []
    log_D_loss = []

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


    # Creation of generator
    N_input_generator = 4
    N_output_generator = 3
    G = LSTM(num_features=N_output_generator * n_DOFs, input_size=N_input_generator*n_DOFs, hidden_size=hidden_size, num_layers=num_layers, seq_length=sequence_len)
    G = torch.nn.DataParallel(G).cuda()
    optimizer = torch.optim.Adam(G.parameters(), lr=args.rate)
    logging.info(repr(optimizer))
    ncpus = os.cpu_count()
    simu_set = TorqueTrackingDataset(N_input_generator,n_DOFs,sequence_len, os.path.join(args.dataset, 'simu.txt'), is_train=True)
    simu_data = DataLoader(simu_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
    logging.info("simulation data: %d batches of batch size %d", len(simu_data), args.batch_size)
    

    # Creation of discriminator
    N_input_discriminator = 3
    N_output_discriminator = 1
    D = LSTM(num_features=N_output_discriminator, input_size=N_input_discriminator*n_DOFs, hidden_size=hidden_size, num_layers=num_layers, seq_length=sequence_len)
    D = torch.nn.DataParallel(D).cuda()
    optimizer = torch.optim.Adam(D.parameters(), lr=args.rate)
    logging.info(repr(optimizer))
    ncpus = os.cpu_count()
    # Set train and valid dataloaders
    real_set = TorqueTrackingDataset(N_input_generator,n_DOFs,sequence_len, os.path.join(args.dataset, 'real.txt'), is_train=True)
    real_data = DataLoader(real_set, batch_size=args.batch_size, shuffle=True, num_workers=ncpus, drop_last=True)
    logging.info("real data: %d batches of batch size %d", len(real_data), args.batch_size)

    # def criterion(output, label) :
    #     output = torch.sigmoid(output)
    #     if label == 0 :
    #         return torch.log(output).nansum()
    #     else : 
    #         return torch.log(torch.ones(hidden_size).cuda()-output).nansum()

    criterion = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(args.epochs):
        if args.outdir and epoch%args.checkpoint==0:
            torch.save(G.state_dict(), os.path.join(args.outdir, "trained_generator_" + str(int(epoch)) + ".model"))
            torch.save(D.state_dict(), os.path.join(args.outdir, "trained_discriminator_" + str(int(epoch)) + ".model"))



        D_loss = train_D(D,G, simu_data, real_data, criterion, optimizer)
        G_loss = train_G(G,D, simu_data, criterion, optimizer)
        log_G_loss.append(G_loss.item())
        log_D_loss.append(D_loss.item())
        logging.info("iters: %d generator_loss: %f discriminator_loss: %f lr: %f",
                     epoch,
                     G_loss.item(),
                     D_loss.item(),
                     optimizer.param_groups[0]['lr'])
        if epoch%10==0:
            lr_scheduler.step()

    torch.save(G.state_dict(), os.path.join(args.outdir, "trained_generator.model"))
    torch.save(G, os.path.join(args.outdir, "trained_generator.pth"))
    torch.save(D.state_dict(), os.path.join(args.outdir, "trained_discriminator.model"))
    torch.save(D, os.path.join(args.outdir, "trained_discriminator.pth"))

    fig = plt.figure()
    plt.plot(log_G_loss,color='teal',linewidth=2,label='Generator Loss')
    plt.plot(log_D_loss,color='lightsalmon',linewidth=2,label='Discriminator Loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('Model training performances')
    plt.show()

if __name__=='__main__':
    main()

