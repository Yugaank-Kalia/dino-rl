# train.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack

# Assuming your environment file is in a folder named 'envs'
from envs.dino_env import DinoEnv

# --- CONFIGURATION ---
LOGDIR = 'runs'
TOTAL_TIMESTEPS = 1_000_000 # Critical: More training time is essential
os.makedirs(LOGDIR, exist_ok=True)

# --- ENVIRONMENT SETUP ---
def make_env():
    """Utility function for env creation"""
    env = DinoEnv()
    env = Monitor(env) # Wraps the env to log rewards and episode lengths
    return env

# Create a vectorized environment
env = DummyVecEnv([make_env])
# Wrap it for frame stacking
env = VecFrameStack(env, n_stack=4)

# --- MODEL AND TRAINING ---
# These are hyperparameters fine-tuned for Atari-like games
model = DQN(
    'CnnPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=50_000,          # Size of the replay buffer
    learning_starts=10_000,      # How many steps to take before starting to learn
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=(4, "step"),      # Train the model every 4 steps
    target_update_interval=1000, # Update the target network every 1000 steps
    exploration_fraction=0.1,    # Fraction of training to explore
    exploration_final_eps=0.01,  # Final value of epsilon
    verbose=1,
    tensorboard_log=LOGDIR,
    device='cuda'
)

# --- CALLBACKS ---
# Save a checkpoint of the model every 50,000 steps
checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=LOGDIR, name_prefix='dino_model')

print(f"🚀 Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("   This will take a significant amount of time. Grab a coffee! ☕")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True # Show a nice progress bar
    )
    model.save(os.path.join(LOGDIR, 'dino_model_final'))
finally:
    env.close()
    print("✅ Training finished!")

# --- PLOTTING (Optional) ---
# You can use TensorBoard to view the learning curve live.
# Open a new terminal and run: tensorboard --logdir runs/DQN_Final