import random
import sys

import pygame

# Initialize Pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bow and Arrow")

# Define colors
BACKGROUND_COLOR = (0, 197, 22)  # RGB value for #00C516
WHITE = (255, 255, 255)

# Load images
character_img = pygame.image.load("character_small.png")
balloon_img = pygame.image.load("balloon_small.png")
yellow_balloon_img = pygame.image.load("yellow_balloon_small.png")
arrow_img = pygame.image.load("arrow_small.png")
red_balloon_falling_img = pygame.image.load("red_balloon_falling_small.png")
yellow_balloon_falling_img = pygame.image.load("yellow_balloon_falling_small.png")

# Set up game variables
character_y = HEIGHT // 2
character_speed = 5
arrow_speed = 3
max_arrows = 20
arrows = []
balloon_speed = 1
max_balloons = 15
max_yellow_balloons = 3
balloons = []
yellow_balloons = []
score = 0
arrows_used = 0
balloons_used = 0
yellow_balloons_used = 0

balloons_hit = []
yellow_balloons_hit = []
red_balloon_falling_speed = 3
yellow_balloon_falling_speed = 3


# Set up fonts
font = pygame.font.Font(None, 36)


# Function to spawn a new balloon
def spawn_balloon():
    x = random.randint(150, WIDTH - 50)
    y = HEIGHT + 50
    balloons.append([x, y])


# Function to spawn a new yellow balloon
def spawn_yellow_balloon():
    x = random.randint(150, WIDTH - 50)
    y = HEIGHT + 50
    if len(balloons) > 0:
        while x not in [b[0] for b in balloons] and y not in [b[1] for b in balloons]:
            x = random.randint(150, WIDTH - 50)
            y = HEIGHT + 50
    yellow_balloons.append([x, y])

# Function to spawn red falling balloon
def spawn_red_falling(balloon):
    x = balloon[0]
    y = balloon[1]
    balloons_hit.append([x, y])

# Function to spawn yellow falling balloon
def spawn_yellow_falling(balloon):
    x = balloon[0]
    y = balloon[1]
    yellow_balloons_hit.append([x, y])

# Main game loop
clock = pygame.time.Clock()
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and arrows_used < max_arrows:
                # Shoot arrow when spacebar is pressed and arrow count is less than max_arrows
                arrow_x = 55
                arrow_y = (character_y + character_img.get_height() // 2) - 10
                arrows.append([arrow_x, arrow_y])
                arrows_used += 1

    # Move the character
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        character_y -= character_speed
    if keys[pygame.K_DOWN]:
        character_y += character_speed

    # Ensure character stays within bounds
    character_y = max(0, min(character_y, HEIGHT - character_img.get_height()))

    # Spawn a new balloon randomly
    if balloons_used < max_balloons and random.randint(1, 300) == 1:
        spawn_balloon()
        balloons_used += 1

    # Spawn a new balloon randomly
    if yellow_balloons_used < max_yellow_balloons and random.randint(1, 500) == 1:
        spawn_yellow_balloon()
        yellow_balloons_used += 1

    # Move the balloons
    for balloon in balloons:
        balloon[1] -= balloon_speed

    # Move the yellow balloons
    for yellow_balloon in yellow_balloons:
        yellow_balloon[1] -= balloon_speed

    # Move the arrows
    for arrow in arrows:
        arrow[0] += arrow_speed

    # Move the red balloons falling
    for balloon in balloons_hit:
        balloon[1] += red_balloon_falling_speed

    # Move the yellow balloons falling
    for yellow_balloon in yellow_balloons_hit:
        yellow_balloon[1] += yellow_balloon_falling_speed

    # Handle collision between arrows and balloons
    for arrow in arrows[:]:
        arrow_rect = arrow_img.get_rect(topleft=(arrow[0], arrow[1]))
        arrow_rect[0] = arrow_rect[0] + 0.9 * arrow_rect[2]
        arrow_rect[2] = 0.1 * arrow_rect[2]
        for balloon in balloons[:]:
            balloon_rect = balloon_img.get_rect(topleft=(balloon[0], balloon[1]))
            balloon_rect[3] = 0.725 * balloon_rect[3]
            if arrow_rect.colliderect(balloon_rect):
                # arrows.remove(arrow)
                if balloon in balloons:
                    spawn_red_falling(balloon)
                    balloons.remove(balloon)
                score += 1

        for yellow_balloon in yellow_balloons[:]:
            yellow_balloon_rect = yellow_balloon_img.get_rect(topleft=(yellow_balloon[0], yellow_balloon[1]))
            yellow_balloon_rect[3] = 0.725 * yellow_balloon_rect[3]
            if arrow_rect.colliderect(yellow_balloon_rect):
                # arrows.remove(arrow)
                if yellow_balloon in yellow_balloons:
                    spawn_yellow_falling(yellow_balloon)
                    yellow_balloons.remove(yellow_balloon)
                score -= 1

    # Draw everything
    screen.fill(BACKGROUND_COLOR)
    screen.blit(character_img, (0, character_y))
    for balloon in balloons:
        if balloon[1] < 0 and random.randint(1, 300) == 1:
            balloon[1] = HEIGHT + 50
        screen.blit(balloon_img, (balloon[0], balloon[1]))

    for yellow_balloon in yellow_balloons:
        if yellow_balloon[1] < 0 and random.randint(1, 500) == 1:
            yellow_balloon[1] = HEIGHT + 50
        screen.blit(yellow_balloon_img, (yellow_balloon[0], yellow_balloon[1]))

    for red_balloon in balloons_hit:
        screen.blit(red_balloon_falling_img,(red_balloon[0],red_balloon[1]))

    for yellow_balloon in yellow_balloons_hit:
        screen.blit(yellow_balloon_falling_img,(yellow_balloon[0],yellow_balloon[1]))

    for arrow in arrows:
        screen.blit(arrow_img, (arrow[0], arrow[1]))

    # Draw the score
    score_text = pygame.font.Font(None,30).render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    # Draw the arrow count
    arrow_text = pygame.font.Font(None,30).render(f"Arrows: {max_arrows - arrows_used}", True, WHITE)
    screen.blit(arrow_text, (10, 40))

    # Check for game over conditions
    if len(balloons) == 0 and balloons_used == 15:
        game_over_text = pygame.font.Font(None,30).render("You WIN!! :)", True, WHITE)
        screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2))

    elif arrows_used == max_arrows:
        game_over_text = pygame.font.Font(None,30).render("Game Over :(", True, WHITE)
        screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
