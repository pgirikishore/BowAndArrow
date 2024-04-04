import random
import sys
from itertools import chain

import gym
import numpy as np
import pygame
from gym import spaces


class BowAndArrowEnv(gym.Env):
    def __init__(self):
        super(BowAndArrowEnv, self).__init__()

        # Initialize Pygame
        pygame.init()

        # Set up the screen
        self.WIDTH, self.HEIGHT = 600, 400
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Bow and Arrow")

        # Define colors
        self.BACKGROUND_COLOR = (0, 197, 22)  # RGB value for #00C516
        self.WHITE = (255, 255, 255)

        # Load images
        self.character_img = pygame.image.load("../character_small.png")
        self.balloon_img = pygame.image.load("../balloon_small.png")
        self.arrow_img = pygame.image.load("../arrow_small.png")

        # Set up game variables
        self.character_y = self.HEIGHT // 2
        self.character_speed = 5
        self.arrow_speed = 5  # 3
        self.max_arrows = 30
        self.arrows = []
        self.balloon_speed = 10
        self.max_balloons = 15
        self.balloons = []
        self.score = 0
        self.arrows_used = 0
        self.balloons_used = 0
        self.curr_reward = 0
        self.previousActions = []

        self.balloons_hit = []

        # Set up fonts
        self.font = pygame.font.Font(None, 36)

        self.action_space = spaces.Discrete(2)  # 0: Do nothing, 1: Shoot arrow
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.WIDTH, self.HEIGHT, 3), dtype=np.uint8)

        self.clock = pygame.time.Clock()
        self.running = True

    def reset(self):
        # Reset game state
        self.previousActions = []
        self.character_y = 0
        self.arrows = []
        self.balloons = []
        self.score = 0
        self.arrows_used = 0
        self.balloons_used = 0

        return self._get_observation()

    def step(self, action):
        self.previousActions.append(action)
        # Move the character within bounds
        self.character_y = (self.HEIGHT - 0) / 2

        # Spawn new balloons randomly
        self._spawn_balloon()

        # Take action
        if action == 1:  # Shoot arrow
            if self.arrows_used < self.max_arrows:
                # Shoot arrow
                arrow_x = 55
                arrow_y = (self.character_y + self.character_img.get_height() // 2) - 10
                self.arrows.append([arrow_x, arrow_y])
                self.arrows_used += 1


            # Move the balloons except the last balloon
            for balloon_index, balloon in enumerate(self.balloons):
                # Check if the balloon is not the last balloon
                if balloon_index != len(self.balloons) - 1:
                    balloon[1] -= self.balloon_speed

            # Move the arrows except the last arrow and remove if they reach the end of the screen
            for arrow_index, arrow in enumerate(self.arrows):
                # Check if the arrow is the last arrow
                if arrow_index != len(self.arrows) - 1:
                    arrow[0] += self.arrow_speed
                    # Remove if the arrow reaches the end of the screen
                    if arrow[0] > self.WIDTH:
                        self.arrows.remove(arrow)

            # Handle collisions
            self._handle_collisions()

            # Calculate current reward
            if self.score - self.curr_reward > 0:
                reward = self.score - self.curr_reward
                self.curr_reward = self.score
            elif self.score - self.curr_reward == 0:
                reward = -1

        if action == 0: # do nothing
            reward = 0

            # Move the balloons
            for balloon in self.balloons:
                balloon[1] -= self.balloon_speed

            # Move the arrow Remove the arrow if it reaches the end of the screen
            for arrow in self.arrows:
                arrow[0] += self.arrow_speed
                if arrow[0] > self.WIDTH:
                    self.arrows.remove(arrow)

            # Handle collisions
            self._handle_collisions()

        # Get observation, reward, done, info
        observation = self._get_observation()
        done = (self.arrows_used == self.max_arrows and len(self.arrows) == 0) or self.score == 15
        info = {}

        return observation, reward, done, info

    def render(self):
        # Draw everything
        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.blit(self.character_img, (0, self.character_y))
        for balloon in self.balloons:
            self.screen.blit(self.balloon_img, (balloon[0], balloon[1]))
        for arrow in self.arrows:
            self.screen.blit(self.arrow_img, (arrow[0], arrow[1]))

        # Display score
        score_surface = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_surface, (0, 380))

        pygame.display.flip()

    def _spawn_balloon(self):
        if self.balloons_used < self.max_balloons and random.randint(1, 100) == 1:
            x = random.randint(150, self.WIDTH - 50)
            y = self.HEIGHT - 10
            self.balloons.append([x, y])
            self.balloons_used += 1


    def _handle_collisions(self):
        # Handle collision between arrows and balloons
        for arrow in self.arrows[:]:
            arrow_rect = pygame.Rect(arrow[0], arrow[1], self.arrow_img.get_width(), self.arrow_img.get_height())
            for balloon in self.balloons[:]:
                balloon_rect = pygame.Rect(balloon[0], balloon[1],
                                           self.balloon_img.get_width(),
                                           self.balloon_img.get_height())
                if arrow_rect.colliderect(balloon_rect):
                    self.balloons.remove(balloon)
                    self.score += 1  # Add score for hitting red balloons
                    break



    def _get_observation(self):
        # Convert game screen to observation
        observation = list(chain.from_iterable(self.balloons)) + list(chain.from_iterable(
            self.arrows)) + list(self.previousActions)
        return observation

    def close(self):
        pygame.quit()
        sys.exit()

    def seed(self, seed=None):
        # Optionally implement seeding logic here
        pass


# Create instance of the custom environment
env = BowAndArrowEnv()

# Test the environment
observation = env.reset()
env.render()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            break

    action = env.action_space.sample()  # Random action
    observation, reward, done, info = env.step(action)
    env.render()

    env.clock.tick(60)
    if done:
        break
