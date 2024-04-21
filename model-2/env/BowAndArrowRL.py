# import os
# import tensorboard
#
# from stable_baselines3 import PPO, DQN, HER
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecFrameStack
#
# from env.BowAndArrowEnv import BowAndArrowEnv
#
# models_dir = "models/PPO"
# logdir = "log"
#
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
#
# if not os.path.exists(logdir):
#     os.makedirs(logdir)
#
#
# # Create a function to make multiple instances of the environment
# def make_env():
#     return BowAndArrowEnv()
#
#
# # Make a vectorized environment
# env = make_vec_env(make_env, n_envs=2)
#
# # Optionally, wrap the environment in a frame stack wrapper
# env = VecFrameStack(env, n_stack=2)
#
# TIMESTEPS = int(1e5)
#
# # Create the PPO model
# model = PPO("CnnPolicy", env, verbose=1, device="mps", tensorboard_log=logdir)
#
# for i in range(1, 30):
#     # Create the PPO model
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#
#     # Save the trained model
#     model.save(f"{models_dir}DQN/{TIMESTEPS * i}")
#
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from BowAndArrowEnv import BowAndArrowEnv  # Ensure this is correctly imported
#
#
# class CNNPolicy(nn.Module):
#     def __init__(self, input_shape, num_actions=2):  # Assuming binary actions: shoot (1) or not shoot (0)
#         super(CNNPolicy, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.fc1_input_size = self._get_conv_output(input_shape)
#         self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=num_actions)  # Outputs for 2 actions
#
#     def _get_conv_output(self, shape):
#         with torch.no_grad():
#             input = torch.rand(1, *shape)
#             output = self.pool1(F.relu(self.conv1(input)))
#             output = self.pool2(F.relu(self.conv2(output)))
#             output = self.pool3(F.relu(self.conv3(output)))
#             return int(np.prod(output.size()))
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
#
#     def act(self, state, epsilon):
#         with torch.no_grad():  # Ensure no gradient computation for inference
#             if torch.rand(1) < epsilon:
#                 # Exploration: Choose a random action
#                 random = torch.randint(0, self.fc2.out_features, (1,)).item()
#                 return random
#             else:
#                 # Preprocess the state to match the expected input shape [N, C, H, W]
#                 state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension: [1, H, W, C]
#                 state = state.permute(0, 1, 2, 3)  # Permute to [1, C, H, W]
#
#                 # Forward pass through the network
#                 q_values = self.forward(state)
#                 # Select the action with the highest Q-value
#                 action = q_values.max(1)[1].item()
#                 return action
#
#
# def preprocess(obs):
#     return obs.transpose((2, 0, 1))  # Convert to CHW format expected by PyTorch
#
#
# def train(env, policy_net, episodes, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
#           epsilon_decay=500):
#     optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
#     losses = []
#     rewards = []
#     epsilon = epsilon_start
#
#     for episode in range(episodes):
#         state = preprocess(env.reset())
#         total_reward = 0
#         done = False
#
#         while not done:
#             action = policy_net.act(state, epsilon)
#             next_state, reward, done, _ = env.step(action)
#             # print(f"Action: {action}, Reward: {reward}")
#             next_state = preprocess(next_state)
#
#             total_reward += reward
#
#             current_q = policy_net(torch.FloatTensor(state).unsqueeze(0)).gather(1, torch.tensor([[action]]))
#             next_q = policy_net(torch.FloatTensor(next_state).unsqueeze(0)).max(1)[0]
#             expected_q = torch.FloatTensor([reward]) + gamma * next_q * (1 - int(done))
#
#             loss = F.smooth_l1_loss(current_q, expected_q.unsqueeze(1))
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             state = next_state
#             epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
#
#         losses.append(loss.item())
#         rewards.append(total_reward)
#         print(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}')
#
#     return rewards, losses
#
#
# # Initialize environment and policy
# env = BowAndArrowEnv()  # Ensure this environment is implemented correctly
# policy_net = CNNPolicy(input_shape=(3, env.HEIGHT, env.WIDTH), num_actions=2)  # Binary action space assumed
#
# # Train the agent
# episodes = 5000
# rewards, losses = train(env, policy_net, episodes)
#
# # Plot training progress
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(rewards)
# plt.title('Rewards')
# plt.subplot(1, 2, 2)
# plt.plot(losses)
# plt.title('Losses')
# plt.show()
#
# env.close()


import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from BowAndArrowEnv import BowAndArrowEnv


class CNNPolicy(nn.Module):
    def __init__(self, input_shape, num_actions=2, dropout_rate=0.5):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm layer for conv1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm layer for conv2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)  # BatchNorm layer for conv3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1_input_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.pool1(F.relu(self.conv1(input)))
            output = self.pool2(F.relu(self.conv2(output)))
            output = self.pool3(F.relu(self.conv3(output)))
            return int(np.prod(output.size()))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Apply BatchNorm after conv1
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))  # Apply BatchNorm after conv2
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))  # Apply BatchNorm after conv3
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        action_scores = self.fc2(x)
        return F.log_softmax(action_scores, dim=1)


def preprocess(obs):
    return (obs.transpose((2, 0, 1)) / 255.0).astype(np.float32)


def train(env, policy_net, episodes, learning_rate=1e-2, gamma=0.99, start_eps=1.0, end_eps=0.01, eps_decay=200):
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-4,
                                                     verbose=True)

    epsilon = start_eps
    steps_done = 0
    reward_threshold = -20  # Set based on environment specifics
    no_improve_in_episodes = 100
    best_average_reward = float('-inf')
    no_improve_count = 0

    for episode in range(episodes):
        saved_log_probs = []
        rewards = []
        state = preprocess(env.reset())
        done = False
        while not done:
            steps_done += 1
            epsilon = end_eps + (start_eps - end_eps) * math.exp(-1. * steps_done / eps_decay)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = torch.exp(policy_net(state_tensor))
            if random.random() > epsilon:
                action_probs = torch.exp(policy_net(state_tensor))
                action = action_probs.max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(2)]], dtype=torch.long)
            saved_log_probs.append(torch.log(action_probs[0][action]))
            state, reward, done, _ = env.step(action.item())
            state = preprocess(state)
            rewards.append(reward)

        # Calculate the returns
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Calculate the policy gradient
        policy_loss = []
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if episode % no_improve_in_episodes == 0:
            recent_average_reward = np.mean(rewards[-no_improve_in_episodes:])
            if recent_average_reward > best_average_reward:
                best_average_reward = recent_average_reward
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count > no_improve_in_episodes:
                print("Stopping early due to no improvement")
                break

        scheduler.step(policy_loss)
        rewards = np.clip(rewards, -1, 10)

        f = open("output.txt", "a")
        f.write(f'Episode {episode}, Rewards: {rewards}, Total Reward: {sum(rewards)}' + "\n")
        f.close()
        print(f'Episode {episode}, Total Reward: {sum(rewards)}')

        # Save the model every 1000 episodes
        if (episode + 1) % 500 == 0:
            model_path = f'policy_net_{episode + 1}.pth'
            torch.save(policy_net.state_dict(), model_path)
            print(f'Model saved at {model_path}')


#Initialize environment and policy
env = BowAndArrowEnv()  # Make sure this environment is correctly implemented
# policy_net = CNNPolicy(input_shape=(1, env.HEIGHT//2, env.WIDTH//2), num_actions=2)
policy_net = CNNPolicy(input_shape=(3, env.HEIGHT, env.WIDTH), num_actions=2)

# Train the agent
episodes = 5000
train(env, policy_net, episodes)