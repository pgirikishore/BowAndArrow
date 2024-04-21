# import math
# import random
# import sys
#
# import gym
# import numpy as np
# import pygame
# from gym import spaces
# import cv2
#
#
# class BowAndArrowEnv(gym.Env):
#     def __init__(self):
#         super(BowAndArrowEnv, self).__init__()
#
#         # Initialize Pygame
#         pygame.init()
#
#         # Set up the screen
#         self.WIDTH, self.HEIGHT = 600, 400
#         self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
#         pygame.display.set_caption("Bow and Arrow")
#
#         # Define colors
#         self.BACKGROUND_COLOR = (0, 197, 22)  # RGB value for #00C516
#         self.WHITE = (255, 255, 255)
#
#         # Load images
#         self.character_img = pygame.image.load("../character_small.png")
#         self.balloon_img = pygame.image.load("../balloon_small.png")
#         self.yellow_balloon_img = pygame.image.load("../yellow_balloon_small.png")
#         self.arrow_img = pygame.image.load("../arrow_small.png")
#         self.red_balloon_falling_img = pygame.image.load("../red_balloon_falling_small.png")
#         self.yellow_balloon_falling_img = pygame.image.load("../yellow_balloon_falling_small.png")
#
#         # Set up game variables
#         self.character_y = 0
#         self.character_speed = 5
#         self.arrow_speed = 25  # 3
#         self.max_arrows = 20
#         self.arrows = []
#         self.balloon_speed = 10 # 1
#         self.max_balloons = 15
#         # self.max_yellow_balloons = 3
#         self.balloons = []
#         # self.yellow_balloons = []
#         self.score = 0
#         self.arrows_used = 0
#         self.balloons_used = 0
#         # self.yellow_balloons_used = 0
#         self.curr_reward = 0
#
#         self.balloons_hit = []
#         # self.yellow_balloons_hit = []
#         self.red_balloon_falling_speed = 3
#         # self.yellow_balloon_falling_speed = 3
#
#         # Set up fonts
#         self.font = pygame.font.Font(None, 36)
#
#         self.action_space = spaces.Discrete(2)  # 0: Do nothing, 1: Shoot arrow
#         self.observation_space = spaces.Box(low=0, high=255, shape=(self.WIDTH//2, self.HEIGHT//2, 1), dtype=np.uint8)
#
#         self.clock = pygame.time.Clock()
#         self.running = True
#
#     def reset(self):
#         # Reset game state
#         self.character_y = 0
#         self.arrows = []
#         self.balloons = []
#         # self.yellow_balloons = []
#         self.score = 0
#         self.arrows_used = 0
#         self.balloons_used = 0
#         self.curr_reward = 0
#         # self.yellow_balloons_used = 0
#
#         return self._get_observation()
#
#     def calculate_distance(self, arrow_x, arrow_y, balloon_x, balloon_y):
#         return math.sqrt((arrow_x - balloon_x) ** 2 + (arrow_y - balloon_y) ** 2)
#
#     def check_hit(self, arrow_rect, balloon_rect):
#         return arrow_rect.colliderect(balloon_rect)
#
#     def calculate_trajectory_reward(self):
#         HIT_REWARD = 10
#         NEAR_MISS_REWARD = 7
#         MISS_PENALTY = -1
#
#         # Initialize reward
#         reward = MISS_PENALTY  # Start with the assumption of a miss
#
#         for arrow in self.arrows:
#             arrow_rect = pygame.Rect(arrow[0], arrow[1], self.arrow_img.get_width(), self.arrow_img.get_height())
#
#             for balloon in self.balloons:
#                 balloon_rect = pygame.Rect(balloon[0], balloon[1], self.balloon_img.get_width(),
#                                            self.balloon_img.get_height())
#
#                 # Check for a direct hit
#                 if self.check_hit(arrow_rect, balloon_rect):
#                     return HIT_REWARD  # Return immediately to avoid further calculations
#
#                 # Calculate distance for a near miss
#                 if reward != HIT_REWARD:  # Only check for near misses if no hit has been detected
#                     distance = self.calculate_distance(arrow[0], arrow[1], balloon[0], balloon[1])
#                     if distance < 150:
#                         reward = NEAR_MISS_REWARD  # Update reward for a near miss
#
#         return reward
#
#     def step(self, action):
#         reward = 0
#         # Perform action
#         # if action == 1:
#         #     self.character_y -= self.character_speed
#         # elif action == 2:
#         #     self.character_y += self.character_speed
#         if action == 1:  # Shoot arrow
#             if self.arrows_used < self.max_arrows:
#                 # Shoot arrow
#                 arrow_x = 55
#                 arrow_y = (self.character_y + self.character_img.get_height() // 2) - 10
#                 self.arrows.append([arrow_x, arrow_y])
#                 self.arrows_used += 1
#                 reward = self.calculate_trajectory_reward()
#
#         # Move the character within bounds
#         self.character_y = max(0, min(self.character_y, self.HEIGHT - self.character_img.get_height()))
#
#         # Spawn new balloons randomly
#         self._spawn_balloon()
#
#         # Spawn yellow balloons randomly
#         # self._spawn_yellow_balloon()
#
#         # Move the balloons
#         for balloon in self.balloons:
#             balloon[1] -= self.balloon_speed
#
#
#         # Move the yellow balloons
#         # for yellow_balloon in self.yellow_balloons:
#         #     yellow_balloon[1] -= self.balloon_speed
#
#         # Move the arrows
#         for arrow in self.arrows:
#             arrow[0] += self.arrow_speed
#
#         # Handle collisions
#         self._handle_collisions()
#
#         # Calculate current reward
#         # if self.score - self.curr_reward != 0:
#         #     reward += (self.score - self.curr_reward) * 10
#         #     self.curr_reward = self.score
#
#         # if action == 0:
#         #     reward = 1
#
#         # Get observation, reward, done, info
#         observation = self._get_observation()
#         done = (self.arrows_used == self.max_arrows and len(self.arrows) == 0) or self.score == 15
#         info = {}
#         return observation, reward, done, info
#
#     def render(self):
#         # Draw everything
#         self.screen.fill(self.BACKGROUND_COLOR)
#         self.screen.blit(self.character_img, (0, self.character_y))
#         for balloon in self.balloons:
#             if balloon[1] < 0 and random.randint(1, 100) == 1:
#                 balloon[1] = self.HEIGHT + 50
#             self.screen.blit(self.balloon_img, (balloon[0], balloon[1]))
#
#         # for yellow_balloon in self.yellow_balloons:
#         #     self.screen.blit(self.yellow_balloon_img, (yellow_balloon[0], yellow_balloon[1]))
#         for arrow in self.arrows:
#             self.screen.blit(self.arrow_img, (arrow[0], arrow[1]))
#
#         # Display score
#         score_surface = self.font.render(f"Score: {self.score}", True, self.WHITE)
#         self.screen.blit(score_surface, (10, 375))
#
#         pygame.display.flip()
#
#     def _spawn_balloon(self):
#         if self.balloons_used < self.max_balloons and random.randint(1, 10) == 1:
#             x = random.randint(150, self.WIDTH - 50)
#             y = self.HEIGHT + 50
#             self.balloons.append([x, y])
#             self.balloons_used += 1
#
#
#     # def _spawn_yellow_balloon(self):
#     #     if self.yellow_balloons_used < self.max_yellow_balloons and random.randint(1, 500) == 1:
#     #         x = random.randint(150, self.WIDTH - 50)
#     #         y = self.HEIGHT + 50
#     #         if len(self.balloons) > 0:
#     #             while x not in [b[0] for b in self.balloons] and y not in [b[1] for b in self.balloons]:
#     #                 x = random.randint(150, self.WIDTH - 50)
#     #                 y = self.HEIGHT + 50
#     #         self.yellow_balloons.append([x, y])
#     #         self.yellow_balloons_used += 1
#
#     def _handle_collisions(self):
#         # Handle collision between arrows and yellow balloons
#         for arrow in self.arrows[:]:
#             arrow_rect = pygame.Rect(arrow[0], arrow[1], self.arrow_img.get_width(), self.arrow_img.get_height())
#         #     for yellow_balloon in self.yellow_balloons[:]:
#         #         yellow_balloon_rect = pygame.Rect(yellow_balloon[0], yellow_balloon[1],
#         #                                           self.yellow_balloon_img.get_width(),
#         #                                           self.yellow_balloon_img.get_height())
#         #         if arrow_rect.colliderect(yellow_balloon_rect):
#         #             self.yellow_balloons.remove(yellow_balloon)
#         #             self.arrows.remove(arrow)
#         #             self.score -= 1  # Deduct score for hitting yellow balloons
#         #             break
#
#             for balloon in self.balloons[:]:
#                 balloon_rect = pygame.Rect(balloon[0], balloon[1],
#                                            self.balloon_img.get_width(),
#                                            self.balloon_img.get_height())
#                 if arrow_rect.colliderect(balloon_rect):
#                     self.balloons.remove(balloon)
#                     if arrow in self.arrows:
#                       self.arrows.remove(arrow)
#                     self.score += 1  # Add score for hitting red balloons
#                     break
#
#         # Remove arrows that go off-screen
#         for arrow in self.arrows[:]:
#             if arrow[0] > self.WIDTH:
#                 self.arrows.remove(arrow)
#
#     def _get_observation(self):
#         # Convert game screen to observation
#         # observation = pygame.surfarray.array3d(self.screen)
#         # return observation
#
#         # Convert game screen to observation
#         raw_observation = pygame.surfarray.array3d(self.screen)
#
#         # Convert to grayscale
#         grayscale_observation = cv2.cvtColor(raw_observation, cv2.COLOR_RGB2GRAY)
#
#         # Downscale the observation (example: downscale to half the original size)
#         downscaled_observation = cv2.resize(grayscale_observation, (self.WIDTH // 2, self.HEIGHT // 2))
#
#         # Add an extra dimension to match the expected input shape of convolutional layers
#         downscaled_observation = np.expand_dims(downscaled_observation, axis=-1)
#
#         return downscaled_observation
#
#     def close(self):
#         pygame.quit()
#         sys.exit()
#
#     def seed(self, seed=None):
#         # Optionally implement seeding logic here
#         pass
#
#
# # # Create instance of the custom environment
# env = BowAndArrowEnv()
#
# # # Test the environment
# observation = env.reset()
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             env.close()
#             break
#
#     action = env.action_space.sample()  # Random action
#     observation, reward, done, info = env.step(action)
#     env.render()
#     env.clock.tick(60)
#     if done:
#         break


import random
import sys

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
        self.character_img = pygame.image.load("../../character_small.png")
        self.balloon_img = pygame.image.load("../../balloon_small.png")
        self.yellow_balloon_img = pygame.image.load("../../yellow_balloon_small.png")
        self.arrow_img = pygame.image.load("../../arrow_small.png")
        self.red_balloon_falling_img = pygame.image.load("../../red_balloon_falling_small.png")
        self.yellow_balloon_falling_img = pygame.image.load("../../yellow_balloon_falling_small.png")

        # Set up game variables
        self.character_y = self.HEIGHT // 2
        self.character_speed = 5
        self.arrow_speed = 25  # 3
        self.max_arrows = 20
        self.arrows = []
        self.balloon_speed = 10 # 1
        self.max_balloons = 15
        # self.max_yellow_balloons = 3
        self.balloons = []
        # self.yellow_balloons = []
        self.score = 0
        self.arrows_used = 0
        self.balloons_used = 0
        # self.yellow_balloons_used = 0
        self.curr_reward = 0

        self.balloons_hit = []
        # self.yellow_balloons_hit = []
        self.red_balloon_falling_speed = 3
        # self.yellow_balloon_falling_speed = 3

        # Set up fonts
        self.font = pygame.font.Font(None, 36)

        self.action_space = spaces.Discrete(2)  # 0: Do nothing, 1: Shoot arrow
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.WIDTH, self.HEIGHT, 3), dtype=np.uint8)

        self.clock = pygame.time.Clock()
        self.running = True

    def reset(self):
        # Reset game state
        self.character_y = 0
        self.arrows = []
        self.balloons = []
        # self.yellow_balloons = []
        self.score = 0
        self.arrows_used = 0
        self.balloons_used = 0
        self.curr_reward = 0
        # self.yellow_balloons_used = 0

        return self._get_observation()

    def step(self, action):
        # Perform action
        # if action == 1:
        #     self.character_y -= self.character_speed
        # elif action == 2:
        #     self.character_y += self.character_speed
        if action == 1:  # Shoot arrow
            if self.arrows_used < self.max_arrows:
                # Shoot arrow
                arrow_x = 55
                arrow_y = (self.character_y + self.character_img.get_height() // 2) - 10
                self.arrows.append([arrow_x, arrow_y])
                self.arrows_used += 1

        # Move the character within bounds
        self.character_y = max(0, min(self.character_y, self.HEIGHT - self.character_img.get_height()))

        # Spawn new balloons randomly
        self._spawn_balloon()

        # Spawn yellow balloons randomly
        # self._spawn_yellow_balloon()

        # Move the balloons
        for balloon in self.balloons:
            balloon[1] -= self.balloon_speed

        # Move the yellow balloons
        # for yellow_balloon in self.yellow_balloons:
        #     yellow_balloon[1] -= self.balloon_speed

        # Move the arrows
        for arrow in self.arrows:
            arrow[0] += self.arrow_speed

        # Handle collisions
        self._handle_collisions()

        # Calculate current reward
        if self.score - self.curr_reward != 0:
            reward = (self.score - self.curr_reward) * 10
            self.curr_reward = self.score
        else:
            reward = len(self.arrows) * -1

        # if action == 0:
        #     reward = 1

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
        # for yellow_balloon in self.yellow_balloons:
        #     self.screen.blit(self.yellow_balloon_img, (yellow_balloon[0], yellow_balloon[1]))
        for arrow in self.arrows:
            self.screen.blit(self.arrow_img, (arrow[0], arrow[1]))

        # Display score
        score_surface = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_surface, (10, 10))

        pygame.display.flip()

    def _spawn_balloon(self):
        if self.balloons_used < self.max_balloons and random.randint(1, 10) == 1:
            x = random.randint(150, self.WIDTH - 50)
            y = self.HEIGHT + 50
            self.balloons.append([x, y])
            self.balloons_used += 1

    # def _spawn_yellow_balloon(self):
    #     if self.yellow_balloons_used < self.max_yellow_balloons and random.randint(1, 500) == 1:
    #         x = random.randint(150, self.WIDTH - 50)
    #         y = self.HEIGHT + 50
    #         if len(self.balloons) > 0:
    #             while x not in [b[0] for b in self.balloons] and y not in [b[1] for b in self.balloons]:
    #                 x = random.randint(150, self.WIDTH - 50)
    #                 y = self.HEIGHT + 50
    #         self.yellow_balloons.append([x, y])
    #         self.yellow_balloons_used += 1

    def _handle_collisions(self):
        # Handle collision between arrows and yellow balloons
        for arrow in self.arrows[:]:
            arrow_rect = pygame.Rect(arrow[0], arrow[1], self.arrow_img.get_width(), self.arrow_img.get_height())
        #     for yellow_balloon in self.yellow_balloons[:]:
        #         yellow_balloon_rect = pygame.Rect(yellow_balloon[0], yellow_balloon[1],
        #                                           self.yellow_balloon_img.get_width(),
        #                                           self.yellow_balloon_img.get_height())
        #         if arrow_rect.colliderect(yellow_balloon_rect):
        #             self.yellow_balloons.remove(yellow_balloon)
        #             self.arrows.remove(arrow)
        #             self.score -= 1  # Deduct score for hitting yellow balloons
        #             break

            for balloon in self.balloons[:]:
                balloon_rect = pygame.Rect(balloon[0], balloon[1],
                                           self.balloon_img.get_width(),
                                           self.balloon_img.get_height())
                if arrow_rect.colliderect(balloon_rect):
                    self.balloons.remove(balloon)
                    if arrow in self.arrows:
                      self.arrows.remove(arrow)
                    self.score += 1  # Add score for hitting red balloons
                    break

        # Remove arrows that go off-screen
        for arrow in self.arrows[:]:
            if arrow[0] > self.WIDTH:
                self.arrows.remove(arrow)

    def _get_observation(self):
        # Convert game screen to observation
        observation = pygame.surfarray.array3d(self.screen)
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