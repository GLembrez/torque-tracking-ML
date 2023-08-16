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
                                 (0.,0.3),
                                 (0.,0.3),
                                 (0.01,0.1),
                                 (1,10))
        self.friction.register(self.model,self.data, self.controller)

        self.tau_d_log = []
        self.q_log = []
        self.q_d_log = []
        self.alpha_log = []
        self.alpha_d_log = []
        self.f_log = []
        self.dtau_log = []
        self.fixed_log = []
        self.c_log = []
        self.out_log = []


    def simulate(self,T):
        sampler = 0
        dtau = np.zeros(7)
        f = np.zeros(7)
        if self.visual:
            while True:
                if self.viewer.is_alive:
                    self.controller.input()

                    if sampler == 4:
                        dtau,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                        self.friction.update(e,dtau)
                        f = self.friction.compute_target(self.controller.cmd_tau)
                        self.friction.compute_target_fixed_point(self.controller.cmd_tau)
                        self.update_log(f)
                        sampler = 0

                    self.data.qfrc_applied = self.controller.cmd_tau - dtau                    

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

                if sampler == 4:
                    dtau,e = self.friction.compute(self.controller.cmd_tau,self.friction.e)
                    self.friction.update(e,dtau)
                    f = self.friction.compute_target(self.controller.cmd_tau)
                    self.friction.compute_target_fixed_point(self.controller.cmd_tau)
                    self.update_log(f)
                    sampler = 0

                self.data.qfrc_applied = self.controller.cmd_tau- dtau 

                mujoco.mj_step(self.model, self.data)
                for i in range(7):
                    if self.controller.t[i] > self.controller.T[i]:
                        self.controller.randomize(i)
                if self.friction.t > self.friction.T:
                    self.friction.randomize()
                sampler += 1
            df = self.register_log()
        return df



    def update_log(self,dtau,output=np.zeros(7)):
        self.tau_d_log.append(self.controller.cmd_tau)#+np.random.normal(0,0.5,size = (7,)))
        self.q_log.append(self.data.qpos.copy())
        self.q_d_log.append(self.controller.q_d.copy())
        self.alpha_log.append(self.data.qvel.copy())
        self.alpha_d_log.append(self.controller.alpha_d.copy())
        self.f_log.append(dtau.copy())
        self.dtau_log.append(self.friction.f.copy())
        self.fixed_log.append(self.friction.fixed.copy())
        self.c_log.append(self.data.qfrc_bias.copy())
        self.out_log.append(output.copy())



    def register_log(self):
        df = pd.DataFrame({"tau_d":self.tau_d_log,
                           "q":self.q_log,
                           "alpha":self.alpha_log,
                           "tau_f":self.f_log,
                           "dtau":self.dtau_log,
                           "f_point":self.fixed_log,
                           "c":self.c_log,
                           "out":self.out_log,
                           "alpha_d":self.alpha_d_log,
                           "q_d":self.q_d_log})
        return df


    def plot(self):
        fig = plt.figure()
        for i in range(7):
            ax = fig.add_subplot(7,1,i+1)
            ax.plot([f[i] for f in self.f_log ], label="ideal friction", color='teal')
            # ax.plot([f[i] for f in self.dtau_log ], label="torque error", color = "red")
            ax.plot([f[i] for f in self.fixed_log ], label="friction fixed point", color='lightsalmon')
            # plt.legend()
            # ax2 = ax.twinx()
            # ax2.plot([a[i] for a in self.alpha_log ], label="velocity",color='lightsalmon')
            # ax2.plot([a[i] for a in self.alpha_d_log ], label="desired velocity",color='blue')
            ax.set_ylabel("DOF " + str(i+1))
            plt.legend()
        plt.show()



def main():
    xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/xml/gen3_7dof_mujoco.xml"
    sim = Simulation(xml_path,visual=True)
    sim.simulate(60000)



if __name__ == "__main__":
    main()
