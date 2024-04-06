from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from PreprocessObservation import PreprocessObservation  # Ensure this import is correct
from BowAndArrowEnv import BowAndArrowEnv  # Ensure this import is correct
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F


def make_env():
    env = BowAndArrowEnv()
    env = PreprocessObservation(env, width=84, height=84)  # Apply preprocessing
    return env


env = make_env()
env = DummyVecEnv([lambda: env])  # Vectorize environment
env = VecFrameStack(env, n_stack=4)  # Stack frames


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Define your CNN architecture here
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.cnn_layers(observations)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x


# Use the custom CNN in a PPO policy
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=10000)  # Adjust timesteps as needed
model.save("ppo_bowandarrow")
