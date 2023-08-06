import mujoco
import mujoco_viewer
import numpy as np
from matplotlib import pyplot as plt
from control import Controller
from error import Friction
import pandas as pd
from tqdm import tqdm
import torch

class Simulation():
    def __init__(self,path_model, visual=False):
        self.visual = visual
        self.model = mujoco.MjModel.from_xml_path(path_model)
        self.data = mujoco.MjData(self.model)
        if self.visual:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title='Kinova Gen3 7 dofs') 
        self.controller = Controller()
        self.controller.register_model(self.model, self.data)
        self.friction = Friction((10000,20000),
                                 (1,100),
                                 (1,4),
                                 (0.25,0),
                                 (0.5,0.7),
                                 (0.5,0.7),
                                 (0.01,0.1),
                                 (0,10))
        self.friction.register(self.model,self.data, self.controller)

        self.tau_d_log = []
        self.q_log = []
        self.alpha_log = []
        self.f_log = []
        self.fixed_log = []
        self.c_log = []
        self.out_log = []

    def simulate(self,T):
        sampler = 0
        if self.visual:
            while True:
                if self.viewer.is_alive:
                    self.controller.input()
                    self.data.qfrc_applied = self.controller.cmd_tau

                    if sampler == 5:
                        f,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                        self.friction.update(e,f)
                        self.friction.find_fixed_point()
                        self.update_log(f)
                        sampler = 0

                    mujoco.mj_step(self.model, self.data)
                    self.viewer.render()
                    if self.controller.t > self.controller.T:
                        self.controller.randomize()
                        self.friction.randomize()

                    sampler += 1
                else:
                    self.viewer.close()
                    self.register_log()
                    self.plot()
                    return
        else:
            for _ in tqdm(range(T)):
                self.controller.input()
                self.data.qfrc_applied = self.controller.cmd_tau

                if sampler == 5:
                    f,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                    self.friction.update(e,f)
                    self.friction.find_fixed_point()
                    self.update_log(f)
                    sampler = 0

                mujoco.mj_step(self.model, self.data)
                if self.controller.t > self.controller.T:
                    self.controller.randomize()
                    self.friction.randomize()
                sampler += 1
            df = self.register_log()
        return df
    
    @torch.no_grad()
    def run_model(self,model):
        input = torch.zeros(10,28).cuda()
        target = torch.zeros(10,7).cuda()
        out = torch.zeros(10,7).cuda()
        dtau = np.zeros((7,))
        loss = []
        sampler = 0

        while True:
            if self.viewer.is_alive:
                self.controller.input()

                if sampler == 5:
                    for i in range(9):
                        input[i,:]=input[i+1,:]
                        target[i,:]=target[i+1,:]
                    input[9,:7] = torch.from_numpy(self.data.qpos).cuda()
                    input[9,7:14] = torch.from_numpy(self.data.qvel).cuda()
                    input[9,14:21] = torch.from_numpy(self.data.qfrc_bias).cuda()
                    input[9,21:] = torch.from_numpy(self.controller.cmd_tau).cuda()
                    out = model(input)
                    dtau = out[-1,:].cpu().numpy()
                    f,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                    self.friction.update(e,f)
                    self.friction.find_fixed_point()
                    sampler = 0
                    self.update_log(f,dtau)


                loss.append(np.linalg.norm(dtau-self.friction.fixed))
                # command the actuators. for real behaviour under compensation, add + dtau - self.friction.f
                self.data.qfrc_applied = self.controller.cmd_tau 
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()
                if self.controller.t > self.controller.T:
                    self.controller.randomize()
                    self.friction.randomize()

                sampler += 1
            else:
                self.viewer.close()
                df = self.register_log()
                return df,loss


          
    def update_log(self,f,output=np.zeros(7)):
        self.tau_d_log.append(self.controller.cmd_tau+np.random.normal(0,0.5,size = (7,)))
        self.q_log.append(self.data.qpos.copy())
        self.alpha_log.append(self.data.qvel.copy())
        self.f_log.append(f.copy())
        self.fixed_log.append(self.friction.fixed.copy())
        self.c_log.append(self.data.qfrc_bias.copy())
        self.out_log.append(output)


    def register_log(self):
        df = pd.DataFrame({"tau_d":self.tau_d_log,
                           "q":self.q_log,
                           "alpha":self.alpha_log,
                           "tau_f":self.f_log,
                           "f_point":self.fixed_log,
                           "c":self.c_log,
                           "out":self.out_log})
        return df


    def plot(self):
        fig = plt.figure()
        for i in range(7):
            ax = fig.add_subplot(7,1,i+1)
            ax.plot([f[i] for f in self.f_log ], label="friction", color='teal')
            ax.plot([f[i] for f in self.fixed_log ], label="fixed point", color = "lightsalmon")
            plt.legend()
            # ax2 = ax.twinx()
            # ax2.plot([a[i] for a in self.alpha_log ], label="fixed point",color='red')
            # ax.set_ylabel("DOF " + str(i+1))
            # plt.legend()
        plt.show()



def main():
    xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/xml/gen3_7dof_mujoco.xml"
    sim = Simulation(xml_path,visual=True)
    sim.simulate(60000)



if __name__ == "__main__":
    main()
