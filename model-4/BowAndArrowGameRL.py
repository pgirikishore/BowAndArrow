import random
import pygame
import numpy as np
from collections import deque
from pygame.locals import *

# Initialize Pygame
pygame.init()
# Set up fonts
font = pygame.font.Font(None, 36)
# Define colors
BACKGROUND_COLOR = (0, 197, 22)  # RGB value for #00C516
WHITE = (255, 255, 255)

SPEED = {
    'clock': 60,
    'character': 5,
    'arrow': 5,
    'red_balloon': 1,
    'red_balloon_falling': 3
}

REWARDS = {
    'shoot_arrow': 2,
    'dont_shoot_arrow': 0,
    'arrow_hit_balloons': 50,
    'arrow_miss_balloons': -5,
    'end_game_multiplier': 2,
    'missed_red_balloon': -5,
    'shoot_consecutive_arrows': -2
}

class BowAndArrowGameAI:

    def __init__(self, w=600, h=400):
        self.w = w
        self.h = h

        #init display
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Bow and Arrow")

        # Load images
        self.character_img = pygame.image.load("../character_small.png")
        self.red_balloon_img = pygame.image.load("../balloon_small.png")
        self.arrow_img = pygame.image.load("../arrow_small.png")
        self.red_balloon_falling_img = pygame.image.load("../red_balloon_falling_small.png")
        self.grid = []
        wt, ht = self.w // 10, self.h // 10
        cw, ch = 0, -ht
        for j in range(100):
            if j % 10 == 0:
                cw = 0
                ch += ht
            rect = pygame.Rect(cw, ch, wt, ht)
            cw += wt
            self.grid.append(rect)
        self.reset()

    def reset(self, record=0):
        # Set up game variables
        self.score = 0
        self.record = record
        self.character_y = self.h // 2
        
        self.arrows = deque([])
        self.arrows_used = 0
        self.max_arrows = 20

        self.red_balloons = deque([])
        self.red_balloons_used = 0
        self.max_red_balloons = 15

        self.red_balloons_hit = []
        self.clock = pygame.time.Clock()
        self.frame_iteration = 0
        self.last_arrow_hit = 0

    def get_balloons_grid(self):
        red_balloons_in_grid = [0]*100
        for balloon in self.red_balloons:
            balloon_rect = self.red_balloon_img.get_rect(topleft=(balloon[0], balloon[1]))
            for i in range(len(red_balloons_in_grid)):
                if self.grid[i].colliderect(balloon_rect):
                    red_balloons_in_grid[i] = 1
        return red_balloons_in_grid

    def play_step(self, action):
        # action = [up, down, arrow]
        self.frame_iteration += 1
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        if np.array_equal(action, [0, 1]):
            if self.arrows:
                #reward += self._shoot_arrow()
                reward += REWARDS['shoot_consecutive_arrows']
            else:
                reward += self._shoot_arrow()
        else:
            reward += REWARDS['dont_shoot_arrow']

        # Spawn a new red balloon randomly
        if self.red_balloons_used < self.max_red_balloons and random.randint(1, 300) == 1:
            self._spawn_red_balloon()
            self.red_balloons_used += 1

        # Move the red balloons
        for red_balloon in self.red_balloons:
            red_balloon[1] -= SPEED['red_balloon']
        while self.red_balloons and self.red_balloons[0][1] < 0:
            self.red_balloons.popleft()
            reward += REWARDS['missed_red_balloon']

        # Move the arrows
        for arrow in self.arrows:
            arrow[0] += SPEED['arrow']
        while self.arrows and self.arrows[0][0] > self.w:
            self.arrows.popleft()
            reward += REWARDS['arrow_miss_balloons']

        # Move the red balloons falling
        for red_balloon_hit in self.red_balloons_hit:
            red_balloon_hit[1] += SPEED['red_balloon_falling']

        # Handle collision between arrows and balloons
        for arrow in self.arrows:
            arrow_rect = self.arrow_img.get_rect(topleft=(arrow[0], arrow[1]))
            arrow_rect[0] = arrow_rect[0] + 0.9 * arrow_rect[2]
            arrow_rect[2] = 0.1 * arrow_rect[2]
            to_be_removed = []
            for balloon in self.red_balloons:
                balloon_rect = self.red_balloon_img.get_rect(topleft=(balloon[0], balloon[1]))
                balloon_rect[3] = 0.725 * balloon_rect[3]
                if arrow_rect.colliderect(balloon_rect):
                    if balloon in self.red_balloons:
                        self.red_balloons_hit.append(balloon)
                        to_be_removed.append(balloon)
                    self.score += 1
                    reward += REWARDS['arrow_hit_balloons']
            for b in to_be_removed:
                self.red_balloons.remove(b)

        game_over, r1 = self._is_game_over()
        if game_over:
            reward = r1
            return reward, game_over, self.score

        self._update_ui()
        self.clock.tick(SPEED['clock'])
        return reward, game_over, self.score
    
    def _is_game_over(self):
        if not self.arrows and ((not self.red_balloons and self.red_balloons_used == self.max_red_balloons) or self.arrows_used == self.max_arrows):
            return True, REWARDS['end_game_multiplier'] * (self.score if self.score > 0 else -3)
        
        return False, 0

    def _shoot_arrow(self):
        if self.arrows_used < self.max_arrows:
            arrow_x = 55
            arrow_y = (self.character_y + self.character_img.get_height() // 2) - 10
            self.arrows.append([arrow_x, arrow_y])
            self.arrows_used += 1
            self.last_arrow_hit = self.frame_iteration
            return REWARDS['shoot_arrow']
        else:
            return 0

    # Function to spawn a new red balloon
    def _spawn_red_balloon(self):
        x = random.randint(150, self.w - 50)
        y = self.h + 50
        self.red_balloons.append([x, y])

    def _move_assets(self, assets, axis, asset_speed, direction):
        # axis = 0 for x axis
        # axis = 1 for y axis
        # direction = 1 for forward
        # direction = -1 for backward
        for asset in assets:
            asset[axis] += direction * SPEED[asset_speed]

    def _update_ui(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.screen.blit(self.character_img, (0, self.character_y))
        for red_balloon in self.red_balloons:
            if red_balloon[1] < 0 and random.randint(1, 300) == 1:
                red_balloon[1] = self.h + 50
            self.screen.blit(self.red_balloon_img, (red_balloon[0], red_balloon[1]))

        for red_balloon in self.red_balloons_hit:
            self.screen.blit(self.red_balloon_falling_img,(red_balloon[0],red_balloon[1]))

        for arrow in self.arrows:
            self.screen.blit(self.arrow_img, (arrow[0], arrow[1]))

        # Draw the record
        score_text = pygame.font.Font(None,30).render(f"Record: {self.record}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        # Draw the score
        score_text = pygame.font.Font(None,30).render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 40))
        # Draw the arrow count
        arrow_text = pygame.font.Font(None,30).render(f"Arrows: {self.max_arrows - self.arrows_used}", True, WHITE)
        self.screen.blit(arrow_text, (10, 70))
        pygame.display.flip()


