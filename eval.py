import time
import numpy as np
from stable_baselines3 import PPO
from envs.dino_env import DinoEnv

model_path = 'runs/ppo_dino_final.zip'

env = DinoEnv()
model = PPO.load(model_path, env=env)

obs, _ = env.reset()
for ep in range(10):
    done = False
    total = 0.0
    obs, _ = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(int(action))
        total += r
        env.render()
        time.sleep(0.01)
    print('Episode reward:', total)

env.close()