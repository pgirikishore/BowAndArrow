from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

from PreprocessObservation import PreprocessObservation
from BowAndArrowEnv import BowAndArrowEnv


def make_env():
    env = BowAndArrowEnv(render_mode='human')
    env = PreprocessObservation(env, width=84, height=84)
    return env


env = make_vec_env(lambda: make_env(), n_envs=1)
env = VecFrameStack(env, n_stack=4)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # Computes shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self._get_conv_output(observation_space.shape)

        self.fc_layers = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def _get_conv_output(self, shape):
        input = torch.autograd.Variable(torch.rand(1, *shape))
        output = self.cnn_layers(input)
        n_size = output.data.view(1, -1).size(1)
        return n_size

    def forward(self, observations):
        x = self.cnn_layers(observations)
        # Flatten the output of the CNN layers
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=2.5e-4, batch_size=64, n_steps=2048)

model.learn(total_timesteps=10000)
model.save("ppo_bowandarrow")
