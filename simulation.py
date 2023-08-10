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
        self.friction = Friction(20000,10,   
                                 (3,5),
                                 (0.,0),
                                 (0.,0.2),
                                 (0.,0.2),
                                 (0.01,0.1),
                                 (1,10))
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
        f = np.zeros(7)
        if self.visual:
            while True:
                if self.viewer.is_alive:
                    self.controller.input()

                    
                    f,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                    self.friction.update(e,f)
                    # if sampler == 5:
                    # self.friction.find_fixed_point()
                    #   sampler = 0
                    self.update_log(f)
                    self.data.qfrc_applied = self.controller.cmd_tau - f                    

                    mujoco.mj_step(self.model, self.data)
                    self.viewer.render()
                    for i in range(7):
                        if self.controller.t[i] > self.controller.T[i]:
                            self.controller.randomize(i)
                    if self.friction.t > self.friction.T:
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

                f,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                self.friction.update(e,f)
                self.update_log(f)
                # if sampler == 5:
                #     self.friction.find_fixed_point()
                #     sampler = 0
                self.data.qfrc_applied = self.controller.cmd_tau- f 

                mujoco.mj_step(self.model, self.data)
                for i in range(7):
                    if self.controller.t[i] > self.controller.T[i]:
                        self.controller.randomize(i)
                if self.friction.t > self.friction.T:
                    self.friction.randomize()
                sampler += 1
            df = self.register_log()
        return df

    @torch.no_grad()
    def run_model(self,model):
        input = torch.zeros(30,28).cuda()
        out = torch.zeros(30,7).cuda()
        dtau = np.zeros((7,))
        loss = []
        sampler = 0

        while True:
            if self.viewer.is_alive:
                self.controller.input()
                f,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                self.friction.update(e,f)
                if sampler == 5:
                    for i in range(29):
                        input[i,:]=input[i+1,:]
                    input[29,:7] = torch.from_numpy(self.data.qpos).cuda()
                    input[29,7:14] = torch.from_numpy(self.data.qvel).cuda()
                    input[29,14:21] = torch.from_numpy(self.data.qfrc_bias).cuda()
                    input[29,21:] = torch.from_numpy(self.controller.cmd_tau).cuda()
                    out = model(input)
                    dtau = out.cpu().numpy()
                    # self.friction.find_fixed_point()
                    sampler = 0
                self.update_log(f,dtau)


                loss.append(np.linalg.norm(dtau-self.friction.f))
                # command the actuators. for real behaviour under compensation, add + dtau - self.friction.f
                self.data.qfrc_applied = self.controller.cmd_tau
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()
                for i in range(7):
                    if self.controller.t[i] > self.controller.T[i]:
                        self.controller.randomize(i)
                if self.friction.t > self.friction.T:
                    self.friction.randomize()

                sampler += 1
            else:
                self.viewer.close()
                df = self.register_log()
                return df,loss



    def update_log(self,f,output=np.zeros(7)):
        self.tau_d_log.append(self.controller.cmd_tau)#+np.random.normal(0,0.5,size = (7,)))
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
            # ax.plot([f[i] for f in self.fixed_log ], label="fixed point", color = "red")
            plt.legend()
            ax2 = ax.twinx()
            ax2.plot([a[i] for a in self.alpha_log ], label="velocity",color='lightsalmon')
            ax.set_ylabel("DOF " + str(i+1))
            plt.legend()
        plt.show()



def main():
    xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/xml/gen3_7dof_mujoco.xml"
    sim = Simulation(xml_path,visual=True)
    sim.simulate(60000)



if __name__ == "__main__":
    main()
