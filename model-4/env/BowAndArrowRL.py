import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from BowAndArrowEnv import BowAndArrowEnv

models_dir = "models/PPO"
logdir = "log"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


# Create a function to make multiple instances of the environment
def make_env():
    return BowAndArrowEnv()


# Make a vectorized environment
env = make_vec_env(make_env, n_envs=2)

# Optionally, wrap the environment in a frame stack wrapper
env = VecFrameStack(env, n_stack=2)

TIMESTEPS = int(1e5)

# Create the PPO model
model = PPO("CnnPolicy", env, verbose=1, device="mps", tensorboard_log=logdir)

for i in range(1, 30):
    # Create the PPO model
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    print("hello")
    # Save the trained model
    model.save(f"{models_dir}/{TIMESTEPS * i}")
