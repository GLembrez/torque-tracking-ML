import mujoco
import mujoco_viewer
import numpy as np
from matplotlib import pyplot as plt
from control import Controller
from error import Friction
import pandas as pd

class Simulation():
    def __init__(self,path_model, path_data):
        self.model = mujoco.MjModel.from_xml_path(path_model)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title='Kinova Gen3 7 dofs')
        self.controller = Controller()
        self.controller.register_model(self.model, self.data)
        self.friction = Friction((1000,20000),
                                 (0,100),
                                 (1,4),
                                 (0,0),
                                 (0.2,0.6),
                                 (0.2,0.6),
                                 (0.01,0.1),
                                 (0,20))
        self.friction.register(self.model,self.data, self.controller)
        self.save_path = path_data

        self.tau_d_log = []
        self.q_log = []
        self.alpha_log = []
        self.f_log = []
        self.fixed_log = []
        self.c_log = []

    def simulate(self):
      sampler = 0
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
          
    def update_log(self,f):
        self.tau_d_log.append(self.controller.cmd_tau+np.random.normal(0,0.5,size = (7,)))
        self.q_log.append(self.data.qpos.copy())
        self.alpha_log.append(self.data.qvel.copy())
        self.f_log.append(f.copy())
        self.fixed_log.append(self.friction.fixed.copy())
        self.c_log.append(self.data.qfrc_bias.copy())


    def register_log(self):
        df = pd.DataFrame({"tau_d":self.tau_d_log,
                           "q":self.q_log,
                           "alpha":self.alpha_log,
                           "tau_f":self.f_log,
                           "f_point":self.fixed_log,
                           "c":self.c_log})
        df.to_pickle(self.save_path)


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
    xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/python_controller/xml/gen3_7dof_mujoco.xml"
    save_path = "/home/gabinlembrez/data/Kinova/train.pkl"
    sim = Simulation(xml_path,save_path)
    sim.simulate()



if __name__ == "__main__":
    main()
