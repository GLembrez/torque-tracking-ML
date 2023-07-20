import mujoco
import mujoco_viewer
import numpy as np
from matplotlib import pyplot as plt
from control import Controller
from error import Friction

class Simulation():
    def __init__(self,path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title='Kinova Gen3 7 dofs')
        self.controller = Controller()
        self.controller.register_model(self.model, self.data)
        self.friction = Friction((1000,20000),
                                 (0,100),
                                 (1,4),
                                 (0,0),
                                 (0.1,0.6),
                                 (0.1,0.6),
                                 (0.01,0.1),
                                 (0,20))
        self.friction.register(self.model,self.data, self.controller)

        self.tau_cmd = []
        self.q_log = []
        self.alpha_log = []
        self.f_log = []
        self.fixed_log = []

    def simulate(self):
      sampler = 0
      while True:
          if self.viewer.is_alive:
              q = self.data.qpos
              alpha = self.data.qvel
              self.controller.input()
              tau = self.controller.cmd_tau
              self.data.qfrc_applied = tau

              if sampler == 5:
                f,e = self.friction.compute(tau,self.friction.e)
                self.friction.update(e,f)
                self.friction.find_fixed_point()
                self.tau_cmd.append(tau)
                self.q_log.append(q.copy())
                self.alpha_log.append(alpha.copy())
                self.f_log.append(f.copy())
                self.fixed_log.append(self.friction.fixed.copy())
                sampler = 0

              mujoco.mj_step(self.model, self.data)
              self.viewer.render()
              if self.controller.t > self.controller.T:
                  self.controller.randomize()
                  self.friction.randomize()

              sampler += 1
          else:
              self.viewer.close()
              self.plot()
              return

    def plot(self):
        fig = plt.figure()
        for i in range(7):
            ax = fig.add_subplot(7,1,i+1)
            ax.plot([f[i] for f in self.f_log ], label="friction", color='teal')
            ax.plot([f[i] for f in self.fixed_log ], label="fixed point", color = "lightsalmon")
            # ax2 = ax.twinx()
            # ax2.plot([f[i] for f in self.alpha_log ], label="fixed point",color='lightsalmon')
            ax.set_ylabel("DOF " + str(i+1))
            plt.legend()
        plt.show()



def main():
    xml_path = "/home/gabinlembrez/GitHub/torque-tracking-ML/python_controller/xml/gen3_7dof_mujoco.xml"
    sim = Simulation(xml_path)
    sim.simulate()



if __name__ == "__main__":
    main()
