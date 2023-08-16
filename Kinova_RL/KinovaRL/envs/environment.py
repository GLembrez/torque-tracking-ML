import numpy as np
import mujoco
import mujoco_viewer
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
from control_RL import Controller
from error import Friction

class KinovaEnv(gym.Env):
    metadata = {}

    def __init__(self,path):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype=float)
        self.action_space = spaces.Box(-5, 5, shape=(7,), dtype=float)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
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
        self.obs = np.zeros((28,))
        self.t = 0
        self.T = 2500

    def _get_obs(self):
        self.obs[:7] = self.data.qpos
        self.obs[7:14] = self.data.qvel
        self.obs[14:21] = self.data.qfrc_bias
        self.obs[21:] = self.controller.cmd_tau
        return self.obs

    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options=None):
        self.obs = np.zeros((28,))
        self.t = 0
        observation = self._get_obs()
        info = self._get_info()
        self.data.qpos= np.zeros(7)
        self.data.qvel= np.zeros(7)      
        self.friction.update(np.zeros(7),np.zeros(7))
        self.friction.randomize()
        for i in range(7):   
            self.controller.randomize(i)   

        return observation, info

    def step(self,action):
        terminated = False 
        if (np.abs(self.data.qvel) > 1e-2).all() or self.t > self.T :
            self.reset()
            terminated = True

        #initialize reward
        reward = 0
        # action is the compensation of the friction
        tau = self.controller.cmd_tau + action

        # compute error
        f,e = self.friction.compute(tau,self.friction.e)
        self.friction.update(e,f)

        # take action
        measured_tau = tau - f 
        self.data.qfrc_applied =measured_tau
        mujoco.mj_step(self.model, self.data)

        # compute reward
        for i in range(7):
            if self.data.qvel[i] < 1e-3:
                # strong penalty for not accounting for stiction
                reward -= 1
            # precision penalty
            reward -= np.linalg.norm(self.data.qvel - self.controller.alpha_d)

        # Compute desired torque
        self.controller.input()

        # observe next state
        observation = self._get_obs()
        info = self._get_info()
        self.t += 1

        return observation,reward,terminated,False, info

        
