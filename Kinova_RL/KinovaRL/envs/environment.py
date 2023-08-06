import numpy as np
import mujoco
import mujoco_viewer
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
from control import Controller
from error import Friction

class KinovaEnv(gym.Env):
    metadata = {}

    def __init__(self,path):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(105,), dtype=float)
        self.action_space = spaces.Box(-10, 10, shape=(7,), dtype=float)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.controller = Controller()
        self.controller.register_model(self.model, self.data)
        self.friction = Friction((1000,10000),
                                 (0,100),
                                 (1,4),
                                 (0,0),
                                 (0.1,0.6),
                                 (0.1,0.6),
                                 (0.01,0.1),
                                 (0,20))
        self.friction.register(self.model,self.data, self.controller)
        self.obs = np.zeros((5,21))
        self.t=0

    def _get_obs(self):
        for i in range(4):
            self.obs[i,:] = self.obs[i+1,:]
        self.obs[-1,:7] = self.data.qpos
        self.obs[-1,7:14] = self.data.qvel
        self.obs[-1,14:] = self.controller.cmd_tau
        return self.obs.flatten()

    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options=None):
        self.obs = np.zeros((5,21))
        self.t = 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self,action):
        terminated = False
        if self.controller.t >= self.controller.T :
            # sample random instances of friction parameters
            self.friction.randomize()
            # chose random trajectory in joint space
            self.controller.randomize() 
        
        if self.t == 100: 
            terminated = True
            self.reset()

        #initialize reward
        reward = 0
        for _ in range(5) :
            # perform 5 simulation steps from one action 
            # action is the compensation of the friction
            tau = self.controller.cmd_tau + action

            # compute error
            f,e = self.friction.compute(tau,self.friction.e)
            self.friction.update(e,f)

            # take action
            measured_tau = tau - f 
            self.data.qfrc_applied = measured_tau
            mujoco.mj_step(self.model, self.data)

            # compute reward
            reward += - 1/5 * np.linalg.norm(measured_tau - self.controller.cmd_tau)

            # Compute desired torque
            self.controller.input()

        # observe next state
        observation = self._get_obs()
        info = self._get_info()

        self.t += 1
        return observation,reward,terminated,False, info

        
