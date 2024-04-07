import gymnasium as gym
import cv2
import numpy as np


class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        super(PreprocessObservation, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, observation):
        # Converts to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Resize
        observation = cv2.resize(observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Channel dimension
        observation = np.expand_dims(observation, axis=-1)
        return observation
