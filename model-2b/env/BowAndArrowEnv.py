import random
import sys

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
import os


class BowAndArrowEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, enable_frame_rate=True):
        super(BowAndArrowEnv, self).__init__()
        if render_mode not in self.metadata['render_modes']:
            raise ValueError(
                f"Render mode {render_mode} is not supported. Supported modes: {self.metadata['render_modes']}")

        self.render_mode = render_mode
        self.enable_frame_rate = enable_frame_rate

        pygame.init()

        self.WIDTH, self.HEIGHT = 600, 400
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Bow and Arrow")

        self.BACKGROUND_COLOR = (0, 197, 22)

        base_path = os.path.dirname(__file__)
        image_dir = os.path.join(base_path, '..', '..', 'images')

        # Change the working directory to the image directory
        os.chdir(image_dir)

        # Load images
        self.character_img = pygame.image.load("character_small.png")
        self.balloon_img = pygame.image.load("balloon_small.png")
        self.yellow_balloon_img = pygame.image.load("yellow_balloon_small.png")
        self.arrow_img = pygame.image.load("arrow_small.png")
        self.red_balloon_falling_img = pygame.image.load("red_balloon_falling_small.png")
        self.yellow_balloon_falling_img = pygame.image.load("yellow_balloon_falling_small.png")

        self.character_y = self.HEIGHT // 2
        self.arrow_speed = 10
        self.max_arrows = 20
        self.arrows = []
        self.balloon_speed = 1
        self.max_balloons = 15
        self.balloons = []
        self.balloons_hit = []
        self.yellow_balloons_hit = []
        self.red_balloon_falling_speed = 3
        self.yellow_balloon_falling_speed = 3
        self.score = 0
        self.arrows_used = 0

        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.WIDTH, self.HEIGHT, 3), dtype=np.uint8)

        self.reset()

    def reset(self, seed=None, options=None):
        self.seed(seed)
        self.character_y = self.HEIGHT // 2
        self.arrows = []
        self.balloons = []
        self.balloons_hit = []
        self.yellow_balloons_hit = []
        self.score = 0
        self.arrows_used = 0
        info = {}
        return self._get_observation(), info

    def step(self, action):
        # print("Taking action:", action)
        reward = 0
        done = False

        if action == 1 and self.arrows_used < self.max_arrows:
            self.arrows.append([55, self.character_y + (self.character_img.get_height() // 2) - 10])
            self.arrows_used += 1

        self._update_balloons()
        self._update_arrows()
        print(self.arrows_used)

        if self.arrows_used == self.max_arrows:
            done = True
            print("Game over condition reached.")

        self.score += reward

        observation = self._get_observation()
        info = {"arrows_left": self.max_arrows - len(self.arrows), "score": self.score}
        truncated = False

        return observation, reward, done, truncated, info

    def _update_balloons(self):
        for balloon in self.balloons:
            balloon['position'][1] -= self.balloon_speed
            if balloon['position'][1] < 0:
                self.balloons.remove(balloon)
                # print("Balloon reached the top of the screen.")

        self._spawn_balloon()

    def _update_arrows(self):
        for arrow in self.arrows[:]:
            arrow[0] += self.arrow_speed
            arrow_rect = pygame.Rect(arrow[0], arrow[1], self.arrow_img.get_width(), self.arrow_img.get_height())

            for balloon in self.balloons[:]:
                balloon_img = self.balloon_img if balloon['color'] == 'red' else self.yellow_balloon_img
                balloon_rect = pygame.Rect(balloon['position'][0], balloon['position'][1], balloon_img.get_width(),
                                           balloon_img.get_height())

                if arrow_rect.colliderect(balloon_rect):
                    self.balloons.remove(balloon)
                    self.arrows.remove(arrow)
                    if balloon['color'] == 'red':
                        self.score += 1
                        self.balloons_hit.append(balloon)
                        print("Red Balloon hit by arrow.")
                    else:
                        self.score -= 1
                        self.yellow_balloons_hit.append(balloon)
                        print("Yellow Balloon hit by arrow.")
                    break

            if arrow[0] > self.WIDTH:
                self.arrows.remove(arrow)
                # print("Arrow reached end of screen.")

    def _spawn_balloon(self):
        if len(self.balloons) < self.max_balloons and random.randint(1, 50) == 1:
            x = random.randint(150, self.WIDTH - 50)
            y = self.HEIGHT
            color = 'red' if random.randint(0, 1) == 0 else 'yellow'
            balloon = {'position': [x, y], 'color': color}
            self.balloons.append(balloon)

    def _get_observation(self):
        self._render_to_surface()
        observation = pygame.surfarray.array3d(pygame.display.get_surface())
        return observation

    def _render_to_surface(self):
        self.screen.fill(self.BACKGROUND_COLOR)
        for balloon in self.balloons:
            balloon_img = self.balloon_img if balloon['color'] == 'red' else self.yellow_balloon_img
            self.screen.blit(balloon_img, balloon['position'])
        for balloon in self.balloons_hit:
            self.screen.blit(self.red_balloon_falling_img, balloon['position'])
            balloon['position'][1] += self.red_balloon_falling_speed
        for balloon in self.yellow_balloons_hit:
            self.screen.blit(self.yellow_balloon_falling_img, balloon['position'])
            balloon['position'][1] += self.yellow_balloon_falling_speed
        for arrow in self.arrows:
            self.screen.blit(self.arrow_img, arrow)
        score_surface = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surface, (10, 10))

    def render(self, mode='human'):
        print("Rendering game...")
        self._render_to_surface()  # Draw the game elements onto the surface

        if mode == 'rgb_array':
            return np.array(pygame.surfarray.pixels3d(self.screen))
        elif mode == 'human':
            if self.render_mode == 'human':
                pygame.display.flip()  # Update the display
                if self.enable_frame_rate:
                    self.clock.tick(self.metadata['render_fps'])

    def close(self):
        pygame.quit()
        sys.exit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# Running the environment
if __name__ == "__main__":
    env = BowAndArrowEnv()
    observation, _ = env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                break

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        env.clock.tick(60)
        if done:
            break
