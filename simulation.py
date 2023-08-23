import mujoco
import mujoco_viewer
import numpy as np
from matplotlib import pyplot as plt
from control import Controller
from error import Friction
import pandas as pd
from tqdm import tqdm
import argparse

class Simulation():
    def __init__(self,path_model, visual=False):

        # initialize mujoco smulation and viewer
        self.visual = visual
        self.model = mujoco.MjModel.from_xml_path(path_model)
        self.data = mujoco.MjData(self.model)
        if self.visual:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title='Kinova Gen3 7 dofs')

        # initialize the control scheme
        self.controller = Controller()
        self.controller.register_model(self.model, self.data)

        # initialize friction
        self.friction = Friction(20000,10,   
                                 (1,6),
                                 (0.,0.3),
                                 (0.,0.3),
                                 (0.,0.),
                                 (0.01,0.1),
                                 (1,10))
        self.friction.register(self.model,self.data, self.controller)

        # Set up data loggers
        self.tau_d_log = [np.zeros(7)]
        self.q_log = [np.zeros(7)]
        self.q_d_log = [np.zeros(7)]
        self.alpha_log = [np.zeros(7)]
        self.alpha_d_log = [np.zeros(7)]
        self.f_log = [np.zeros(7)]
        self.dtau_log = [np.zeros(7)]
        self.fixed_log = [np.zeros(7)]
        self.c_log = [np.zeros(7)]
        self.out_log = [np.zeros(7)]
        self.Fs_log = [np.zeros(7)]


    def simulate(self,T):
        """
        Executes the mujoco simulation. 2 modes are available

        if self.visual is True: 
        The simulation is executed as long aas the viewer is active

        if self.visual is False:
        The simulation is run for T steps and returns the logs in a dataframe

        * Compute the desired torque
        * Every five iterations:
            * compute joint friction
            * compute ideal friction
            * compute ideal friction fixed point
            * update logs
        * do a step of mujoco with input = desired torque - real friction
        * check if the joints need a new target position
        * check if it is time for domain randomization of friction

        f    - ideal friction 
        dtau - real friction
        df   - dataframe storing all the logs
        """
        sampler = 0
        dtau = np.zeros(7)
        f = np.zeros(7)
        if self.visual:
            while True:
                if self.viewer.is_alive:
                    self.controller.input()

                    if sampler == 4:
                        f = self.friction.compute_target(self.controller.cmd_tau)
                        self.friction.compute_target_fixed_point(self.controller.cmd_tau)
                        dtau,e = self.friction.compute(self.controller.cmd_tau+f,self.friction.e)
                        self.friction.update(e,dtau)
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

                self.data.qfrc_applied = self.controller.cmd_tau - dtau 
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
        self.tau_d_log.append(self.controller.cmd_tau)              # desired torque
        self.q_log.append(self.data.qpos.copy())                    # real position
        self.q_d_log.append(self.controller.q_d.copy())             # desired position
        self.alpha_log.append(self.data.qvel.copy())                # real velocity
        self.alpha_d_log.append(self.controller.alpha_d.copy())     # desired velocity
        self.f_log.append(dtau.copy())                              # real error
        self.dtau_log.append(self.friction.f.copy())                # real optimal friction
        self.fixed_log.append(self.friction.fixed.copy())           # real fixed point of optimal friction
        self.c_log.append(self.data.qfrc_bias.copy())               # coriolis-gravity vector
        self.out_log.append(output.copy())                          # prediction (if exists)
        self.Fs_log.append(self.friction.Fs.copy())                 # stiction torque


    def register_log(self):
        """
        returns the panda dataframe containing all the logs
        df behaves as a dictionnary where each item is a list of numpy arrays of size 7 for each timestep
        """
        df = pd.DataFrame({"tau_d":self.tau_d_log,
                           "q":self.q_log,
                           "alpha":self.alpha_log,
                           "tau_f":self.f_log,
                           "dtau":self.dtau_log,
                           "f_point":self.fixed_log,
                           "c":self.c_log,
                           "out":self.out_log,
                           "alpha_d":self.alpha_d_log,
                           "q_d":self.q_d_log,
                           "Fs":self.Fs_log})
        return df


    def plot(self):
        """
        Plots relevant graphs using pyplot
        can be modified
        """
        fig = plt.figure()
        for i in range(7):
            ax = fig.add_subplot(7,1,i+1)
            ax.plot([f[i] for f in self.dtau_log ], label="torque error", color='teal')
            ax.plot([f[i] for f in self.fixed_log ], label="ideal friction fixed point", color = "lightsalmon")
            plt.legend()
            ax2 = ax.twinx()
            ax2.plot([a[i] for a in self.alpha_log ], '--', label="velocity",color='red')
            ax2.plot([a[i] for a in self.alpha_d_log ], '--', label="desired velocity",color='blue')
            plt.legend()
            ax.set_ylabel("DOF " + str(i+1))
        fig.tight_layout()
        plt.show()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, default=False)
    args = ap.parse_args()
    sim = Simulation(args.path,visual=True)
    sim.simulate(0)



if __name__ == "__main__":
    main()
