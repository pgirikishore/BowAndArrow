from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from BowAndArrowEnv import BowAndArrowEnv  # Adjust the import path as needed
from PreprocessObservation import PreprocessObservation  # Adjust the import path as needed

def make_env():
    env = BowAndArrowEnv()
    env = PreprocessObservation(env, width=84, height=84)  # Same preprocessing as during training
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

model_path = "path_to_your_saved_model"  # Adjust this to your model's file path
model = PPO.load(model_path, env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    env.render()  # Make sure your environment supports rendering
    if dones.any():
        obs = env.reset()