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
        self.arrow_speed = 3
        self.max_arrows = 20
        self.arrows = []
        self.balloon_speed = 1
        self.max_balloons = 15
        self.max_yellow_balloons = 3
        self.balloons = []
        self.yellow_balloons = []
        self.score = 0
        self.arrows_used = 0
        self.balloons_used = 0
        self.yellow_balloons_used = 0

        self.balloons_hit = []
        self.yellow_balloons_hit = []
        self.red_balloon_falling_speed = 3
        self.yellow_balloon_falling_speed = 3

        # Set up fonts
        self.font = pygame.font.Font(None, 36)

        self.action_space = spaces.Discrete(4)  # 0: Do nothing, 1: Move up, 2: Move down, 3: Shoot arrow
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.WIDTH, self.HEIGHT, 3), dtype=np.uint8)

        self.clock = pygame.time.Clock()
        self.running = True

    def reset(self):
        # Reset game state
        self.character_y = self.HEIGHT // 2
        self.arrows = []
        self.balloons = []
        self.yellow_balloons = []
        self.score = 0
        self.arrows_used = 0
        self.balloons_used = 0
        self.yellow_balloons_used = 0

        return self._get_observation()

    def step(self, action):
        # Perform action
        if action == 1:
            self.character_y -= self.character_speed
        elif action == 2:
            self.character_y += self.character_speed
        elif action == 3:  # Shoot arrow
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
        self._spawn_yellow_balloon()

        # Move the balloons
        for balloon in self.balloons:
            balloon[1] -= self.balloon_speed

        # Move the yellow balloons
        for yellow_balloon in self.yellow_balloons:
            yellow_balloon[1] -= self.balloon_speed

        # Move the arrows
        for arrow in self.arrows:
            arrow[0] += self.arrow_speed

        # Handle collisions
        self._handle_collisions()

        # Get observation, reward, done, info
        observation = self._get_observation()
        reward = self.score
        done = self.arrows_used == self.max_arrows
        info = {}

        return observation, reward, done, info

    def render(self):
        # Draw everything
        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.blit(self.character_img, (0, self.character_y))
        for balloon in self.balloons:
            self.screen.blit(self.balloon_img, (balloon[0], balloon[1]))
        for yellow_balloon in self.yellow_balloons:
            self.screen.blit(self.yellow_balloon_img, (yellow_balloon[0], yellow_balloon[1]))
        for arrow in self.arrows:
            self.screen.blit(self.arrow_img, (arrow[0], arrow[1]))

        # Display score
        score_surface = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_surface, (10, 10))

        pygame.display.flip()

    def _spawn_balloon(self):
        if self.balloons_used < self.max_balloons and random.randint(1, 300) == 1:
            x = random.randint(150, self.WIDTH - 50)
            y = self.HEIGHT + 50
            self.balloons.append([x, y])
            self.balloons_used += 1

    def _spawn_yellow_balloon(self):
        if self.yellow_balloons_used < self.max_yellow_balloons and random.randint(1, 500) == 1:
            x = random.randint(150, self.WIDTH - 50)
            y = self.HEIGHT + 50
            if len(self.balloons) > 0:
                while x not in [b[0] for b in self.balloons] and y not in [b[1] for b in self.balloons]:
                    x = random.randint(150, self.WIDTH - 50)
                    y = self.HEIGHT + 50
            self.yellow_balloons.append([x, y])
            self.yellow_balloons_used += 1

    def _handle_collisions(self):
        # Handle collision between arrows and yellow balloons
        for arrow in self.arrows[:]:
            arrow_rect = pygame.Rect(arrow[0], arrow[1], self.arrow_img.get_width(), self.arrow_img.get_height())
            for yellow_balloon in self.yellow_balloons[:]:
                yellow_balloon_rect = pygame.Rect(yellow_balloon[0], yellow_balloon[1],
                                                  self.yellow_balloon_img.get_width(),
                                                  self.yellow_balloon_img.get_height())
                if arrow_rect.colliderect(yellow_balloon_rect):
                    self.yellow_balloons.remove(yellow_balloon)
                    self.arrows.remove(arrow)
                    self.score -= 1  # Deduct score for hitting yellow balloons
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

