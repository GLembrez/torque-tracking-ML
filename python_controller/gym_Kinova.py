import KinovaRL
import gymnasium as gym
import numpy as np

env = gym.make("Kinova_RL/KinovaEnv", path="/home/gabinlembrez/GitHub/torque-tracking-ML/python_controller/xml/gen3_7dof_mujoco.xml")

env.reset()
print(env.step(np.zeros(7)))