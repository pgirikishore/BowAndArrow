from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from BowAndArrowEnv import BowAndArrowEnv


# Create a function to make multiple instances of the environment
def make_env():
    return BowAndArrowEnv()

# Load the trained model
model = PPO.load("models/PPO/100000")

# Make a vectorized environment for evaluation
eval_env = make_vec_env(make_env, n_envs=1)  # You can adjust the number of environments as needed

# Optionally, wrap the environment in a frame stack wrapper if you used it during training
eval_env = VecFrameStack(eval_env, n_stack=2)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Use the trained model for inference
obs = eval_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = eval_env.step(action)
    eval_env.render()
    if done:
        obs = eval_env.reset()  # Reset the environment if the episode is done
