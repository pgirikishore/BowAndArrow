# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.vec_env import VecFrameStack
#
# from env.BowAndArrowEnv import BowAndArrowEnv
#
#
# # Create a function to make multiple instances of the environment
# def make_env():
#     return BowAndArrowEnv()
#
#
# # Load the trained model
# model = PPO.load("models/PPO/100000")
#
# # Make a vectorized environment for evaluation
# eval_env = make_vec_env(make_env, n_envs=1)  # You can adjust the number of environments as needed
#
# # Optionally, wrap the environment in a frame stack wrapper if you used it during training
# eval_env = VecFrameStack(eval_env, n_stack=2)
#
# # # Evaluate the trained model
# # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
# #
# # print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
#
# # Use the trained model for inference
# obs = eval_env.reset()
# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     print(action)
#     obs, reward, done, _ = eval_env.step(action)
#     # eval_env.render()
#     if done[0]:
#         obs = eval_env.reset()  # Reset the environment if the episode is done
import pygame
import torch
import numpy as np
from BowAndArrowEnv import BowAndArrowEnv  # Make sure this is correctly imported
from BowAndArrowRL import CNNPolicy  # Adjust the import according to your file structure

# Initialize the model instance
model_path = 'policy_net_1000.pth'
policy_net = CNNPolicy(input_shape=(1, 300, 200), num_actions=2)  # Adjust the input shape as necessary
policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))  # Ensure compatibility across different devices
policy_net.eval()

def preprocess(obs):
    return (obs.transpose((2, 0, 1)) / 255.0).astype(np.float32)

def play_game(env, policy_net):
    state = env.reset()
    state = preprocess(state)  # Preprocess initial state
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add a batch dimension
        with torch.no_grad():  # Ensuring PyTorch does not track gradients
            action_probs = torch.exp(policy_net(state_tensor))
        print(action_probs)
        action = action_probs.argmax().item()  # Select the action with the highest probability
        state, reward, done, _ = env.step(action)
        state = preprocess(state)  # Preprocess the new state
        env.render()  # Visualize the game
        env.clock.tick(60)  # Control the game's frame rate


if __name__ == "__main__":
    pygame.init()

    WIDTH, HEIGHT = 600, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bow and Arrow")
    # Create an instance of the game environment
    env = BowAndArrowEnv()

    # Play the game with the loaded model
    play_game(env, policy_net)
