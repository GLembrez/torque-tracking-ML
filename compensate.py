import pandas as pd
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
from progress.bar import Bar
import mujoco

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from net import LSTM
from data_loader import TorqueTrackingDataset
from simulation import Simulation


####################### HYPERPARAMETERS ###########################

n_DOFs = 7
input_len = 4
sequence_len = 10
num_layers = 2
hidden_size = 32
tol = 1e-2                  # threshold on the improvement of the valid loss
xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/xml/gen3_7dof_mujoco.xml"

###################################################################

def initialize_net(trained_weights):
    input_len = 28
    sequence_len = 10
    net = LSTM(num_features=7, input_size=input_len, hidden_size=32, num_layers=2, seq_length=sequence_len)
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

def clean_plot(df,loss):

    print(np.average(loss))
    loss_fig = plt.figure()
    plt.plot(loss,color = 'teal', label='compensation error')
    plt.xlabel("epoch")
    plt.ylabel("RMSE [Nm]")

    time_perf = plt.figure()
    for i in range(7):
        ax = time_perf.add_subplot(7,1,i+1)
        pred = [t[i] for t in df["out"]]
        error = [t[i] for t in df["dtau"]]
        ax.plot(pred,color='teal',label="prediction")
        ax.plot(error,color='lightsalmon',label="real error")
        plt.legend()
        ax.set_ylabel("DOF "+str(i+1))
    ax.set_xlabel("iteration")

    plt.show()

@torch.no_grad()
def run_model(sim,model):
    input = torch.zeros(10,28).cuda()
    out = torch.zeros(7).cuda()
    dtau = np.zeros((7,))
    loss = []
    sampler = 0

    while True:
        if sim.viewer.is_alive:
            sim.controller.input()
            f,e = sim.friction.compute(sim.controller.cmd_tau,sim.friction.e)
            sim.friction.update(e,f)
            if sampler == 5:
                X = torch.zeros(1,28).cuda()
                X[-1,:7] = torch.from_numpy(sim.controller.alpha_d).cuda()
                X[-1,7:14] = torch.from_numpy(sim.data.qfrc_bias).cuda()
                X[-1,14:21] = torch.from_numpy(sim.controller.cmd_tau).cuda()
                X[-1,21:] = torch.from_numpy(sim.friction.f).cuda()
                input = torch.cat((input[1:,:],X))
                out = model(input)
                # out,input = fixed_point(model,input)
                dtau = out.cpu().numpy().copy()
                # sim.friction.find_fixed_point()
                sampler = 0
                sim.update_log(f,dtau)


            loss.append(np.linalg.norm(dtau-sim.friction.f))
            # command the actuators. for real behaviour under compensation, add + dtau - sim.friction.f
            sim.data.qfrc_applied = sim.controller.cmd_tau - f
            mujoco.mj_step(sim.model, sim.data)
            sim.viewer.render()
            for i in range(7):
                if sim.controller.t[i] > sim.controller.T[i]:
                    sim.controller.randomize(i)
            if sim.friction.t > sim.friction.T:
                sim.friction.randomize()

            sampler += 1
        else:
            sim.viewer.close()
            df = sim.register_log()
            return df,loss
        
@torch.no_grad()
def fixed_point(model, input):

    out = torch.zeros(7).cuda()
    for i in range(7):
        n_iter = 0
        tol = 1e-2
        epsilon = 1e-3
        lr = 1
        grad = 0
        dtau = model(input)[i]
        F = eval_F(model,input,dtau,i)
        while torch.abs(F)>tol and n_iter < 100: # and exists :   
        # Newton Raphson loop
            F_new = eval_F(model,input,dtau,i)
            grad =  (F_new - F) / epsilon  -1
            dtau -= lr * F/grad
            F = eval_F(model,input,dtau,i)
            n_iter += 1
        out[i] = dtau
    input[-1,21:] += out
    return out,input

@torch.no_grad()
def eval_F(model, input, dtau,i):
    dX = torch.zeros(10,28).cuda()
    dX[-1,21+i] += dtau
    return model(input+dX)[i] - dtau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    required=True,
                    type=str
    )
    args = ap.parse_args()

    net = initialize_net(args.model)
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    sim = Simulation(xml_path,visual=True)
    df,loss = run_model(sim,net)
    clean_plot(df,loss)

    return



if __name__=='__main__':
    main()