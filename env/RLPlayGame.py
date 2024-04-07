from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from BowAndArrowEnv import BowAndArrowEnv
from PreprocessObservation import PreprocessObservation

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

def make_env():
    env = BowAndArrowEnv(render_mode='human')
    # Applying the same preprocessing as during training
    env = PreprocessObservation(env, width=84, height=84)
    return env


env = make_vec_env(lambda: make_env(), n_envs=1)
env = VecFrameStack(env, n_stack=4)

# Load the model if it exists, otherwise initialize a new one
try:
    model = PPO.load("ppo_bowandarrow.zip")
except ValueError:
    model = PPO("CnnPolicy", env, verbose=1)


def evaluate_model(model, eval_env, n_eval_episodes=10):
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Mean Reward: {mean_reward}")

# Total number of episodes for training
total_episodes = 1000
# Evaluates the model every 100 episodes
eval_interval = 100

for episode in range(total_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

    if (episode + 1) % eval_interval == 0:
        print(f"Evaluating at episode {episode + 1}")
        evaluate_model(model, env)  # Evaluate the model

model.save("ppo_bowandarrow_final")
