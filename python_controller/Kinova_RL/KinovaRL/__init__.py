from gymnasium.envs.registration import register

register(
     id="Kinova_RL/KinovaEnv",
     entry_point="KinovaRL.envs:KinovaEnv",
     max_episode_steps=2501,
)