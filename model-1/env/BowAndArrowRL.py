from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch

from BowAndArrowEnv import BowAndArrowEnv

print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Create a function to make multiple instances of the environment
def make_env():
    return BowAndArrowEnv()

# Make a vectorized environment
env = make_vec_env(make_env, n_envs=1)

# Optionally, wrap the environment in a frame stack wrapper
env = VecFrameStack(env, n_stack=1)

# Create and train the PPO model
model = PPO("CnnPolicy", env, verbose=1, device="cuda")


model.learn(total_timesteps=int(1e5))

# Save the trained model
model.save("ppo_bow_and_arrow")
