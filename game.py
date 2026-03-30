import pygame 
import random
import sys
from enum import Enum

pygame.init()

SCREEN_WIDTH=800
SCREEN_HEIGHT=600
BLOCK_SIZE=20
GAME_SPEED=10

class Colors:
    BG_TOP = (15, 32, 39)
    BG_BOTTOM = (44, 83, 100)

    #Snake Colors
    SNAKE_HEAD = (46, 213, 115) 
    SNAKE_BODY_1 = (34, 166, 90) 
    SNAKE_BODY_2 = (25, 130, 70)

    FOOD_COLOR = (252, 92, 101)
    FOOD_GLOW = (255, 159, 165)

    # UI
    TEXT_COLOR = (255, 255, 255) 
    SCORE_BG = (30, 39, 46, 180)     
    GAME_OVER_BG = (44, 62, 80, 230)

    GRID_COLOR = (40, 55, 71)


pygame.font.init()
FONT_LARGE = pygame.font.Font(None, 72)
FONT_MEDIUM = pygame.font.Font(None, 48)
FONT_SMALL = pygame.font.Font(None, 36)

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def draw_gradient_background(screen):
    for y in range(SCREEN_HEIGHT):
        ratio = y / SCREEN_HEIGHT
        
        r = int(Colors.BG_TOP[0] * (1 - ratio) + Colors.BG_BOTTOM[0] * ratio)
        g = int(Colors.BG_TOP[1] * (1 - ratio) + Colors.BG_BOTTOM[1] * ratio)
        b = int(Colors.BG_TOP[2] * (1 - ratio) + Colors.BG_BOTTOM[2] * ratio)
        
        pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

def draw_grid(screen):
    for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
        pygame.draw.line(screen, Colors.GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
    
    for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
        pygame.draw.line(screen, Colors.GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)

def draw_rounded_rect(surface, color, rect, corner_radius):
    pygame.draw.rect(surface, color, rect, border_radius=corner_radius)


class Snake:
    def __init__(self):
        """Initialize snake in the center"""
        # Starting position (center of screen)
        start_x = SCREEN_WIDTH // 2
        start_y = SCREEN_HEIGHT // 2
        
        # Snake body (list of positions)
        self.body = [
            [start_x, start_y],
            [start_x - BLOCK_SIZE, start_y],
            [start_x - (2 * BLOCK_SIZE), start_y]
        ]
        
        self.direction = Direction.RIGHT
        self.grow = False  # Flag to grow snake
    
    def get_head(self):
        """Return head position"""
        return self.body[0]
    
    def move(self, new_direction):
        """Move snake in given direction"""
        # Prevent 180-degree turns
        if new_direction == Direction.UP and self.direction != Direction.DOWN:
            self.direction = new_direction
        elif new_direction == Direction.DOWN and self.direction != Direction.UP:
            self.direction = new_direction
        elif new_direction == Direction.LEFT and self.direction != Direction.RIGHT:
            self.direction = new_direction
        elif new_direction == Direction.RIGHT and self.direction != Direction.LEFT:
            self.direction = new_direction
        
        # Calculate new head position
        head = self.get_head().copy()
        
        if self.direction == Direction.UP:
            head[1] -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            head[1] += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            head[0] -= BLOCK_SIZE
        elif self.direction == Direction.RIGHT:
            head[0] += BLOCK_SIZE
        
        # Add new head
        self.body.insert(0, head)
        
        # Remove tail unless growing
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
    
    def grow_snake(self):
        """Mark snake to grow on next move"""
        self.grow = True
    
    def check_collision(self):
        """Check if snake collided with walls or itself"""
        head = self.get_head()
        
        # Wall collision
        if (head[0] < 0 or head[0] >= SCREEN_WIDTH or
            head[1] < 0 or head[1] >= SCREEN_HEIGHT):
            return True
        
        if head in self.body[1:]:
            return True
        
        return False
    
    def draw(self, screen):
        for i, segment in enumerate(self.body):
            # Head is brightest
            if i == 0:
                color = Colors.SNAKE_HEAD
                # Draw head with slight glow effect
                glow_rect = pygame.Rect(
                    segment[0] - 2, segment[1] - 2,
                    BLOCK_SIZE + 4, BLOCK_SIZE + 4
                )
                pygame.draw.rect(screen, Colors.SNAKE_BODY_1, glow_rect, border_radius=6)
            else:
                if i % 2 == 0:
                    color = Colors.SNAKE_BODY_1
                else:
                    color = Colors.SNAKE_BODY_2
            
            rect = pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, color, rect, border_radius=5)
            
            highlight = pygame.Rect(
                segment[0] + 2, segment[1] + 2,
                BLOCK_SIZE - 10, BLOCK_SIZE - 10
            )
            highlight_color = tuple(min(c + 30, 255) for c in color)
            pygame.draw.rect(screen, highlight_color, highlight, border_radius=3)


class Food:
    def __init__(self):
        """Initialize food at random position"""
        self.position = [0, 0]
        self.pulse = 0  # For pulsing animation
        self.respawn()
    
    def respawn(self, snake_body=None):
        while True:
            self.position = [
                random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            ]
            
            # Make sure food doesn't spawn on snake
            if snake_body is None or self.position not in snake_body:
                break
        
    def draw(self, screen):
        # Pulsing animation
        self.pulse += 0.1
        pulse_size = int(2 * abs(pygame.math.Vector2(1, 0).rotate(self.pulse * 10).x))
        
        # Outer glow
        glow_rect = pygame.Rect(
            self.position[0] - pulse_size - 2,
            self.position[1] - pulse_size - 2,
            BLOCK_SIZE + (pulse_size * 2) + 4,
            BLOCK_SIZE + (pulse_size * 2) + 4
        )
        pygame.draw.rect(screen, Colors.FOOD_GLOW, glow_rect, border_radius=BLOCK_SIZE // 2)
        
        # Main food circle
        food_rect = pygame.Rect(
            self.position[0], self.position[1],
            BLOCK_SIZE, BLOCK_SIZE
        )
        pygame.draw.rect(screen, Colors.FOOD_COLOR, food_rect, border_radius=BLOCK_SIZE // 2)
        
        # Inner shine
        shine_rect = pygame.Rect(
            self.position[0] + 4, self.position[1] + 4,
            BLOCK_SIZE - 12, BLOCK_SIZE - 12
        )
        shine_color = tuple(min(c + 50, 255) for c in Colors.FOOD_COLOR)
        pygame.draw.rect(screen, shine_color, shine_rect, border_radius=BLOCK_SIZE // 2)



class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('🐍 Beautiful Snake Game')
        self.clock = pygame.time.Clock()
        
        # Game objects
        self.snake = Snake()
        self.food = Food()
        self.food.respawn(self.snake.body)
        
        # Game state
        self.score = 0
        self.high_score = 0
        self.game_over = False
        self.paused = False
    
    def reset(self):
        """Reset game to start new round"""
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False
        self.paused = False
    
    def handle_input(self):
        """Process keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if self.game_over:
                    if event.key == pygame.K_SPACE:
                        self.reset()
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                else:
                    # Movement controls
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.snake.move(Direction.UP)
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.snake.move(Direction.DOWN)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.snake.move(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.snake.move(Direction.RIGHT)
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
    
    def update(self):
        if self.game_over or self.paused:
            return
        
        # Move snake
        self.snake.move(self.snake.direction)
        
        # Get head position
        head = self.snake.get_head()
        
        # Check collision with food (compare x and y coordinates)
        if head[0] == self.food.position[0] and head[1] == self.food.position[1]:
            # Food eaten!
            self.score += 2
            self.snake.grow_snake()
            self.food.respawn(self.snake.body)
            
            # Update high score
            if self.score > self.high_score:
                self.high_score = self.score
        
        # Check collision with walls or self
        if self.snake.check_collision():
            self.game_over = True
    
    def draw_ui(self):
        """Draw score and UI elements"""
        # Score panel (top-left)
        panel_width = 250
        panel_height = 80
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill(Colors.SCORE_BG)
        
        # Draw score text
        score_text = FONT_SMALL.render(f'Score: {self.score}', True, Colors.TEXT_COLOR)
        high_score_text = FONT_SMALL.render(f'Best: {self.high_score}', True, Colors.TEXT_COLOR)
        
        panel.blit(score_text, (10, 10))
        panel.blit(high_score_text, (10, 45))
        
        self.screen.blit(panel, (10, 10))
        
        # Pause indicator
        if self.paused:
            pause_text = FONT_LARGE.render('PAUSED', True, Colors.TEXT_COLOR)
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
            # Semi-transparent background
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            self.screen.blit(pause_text, text_rect)
    
    def draw_game_over(self):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Game Over panel
        panel_width = 500
        panel_height = 350
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill(Colors.GAME_OVER_BG)
        
        # Text
        game_over_text = FONT_LARGE.render('GAME OVER', True, Colors.TEXT_COLOR)
        score_text = FONT_MEDIUM.render(f'Final Score: {self.score}', True, Colors.TEXT_COLOR)
        high_score_text = FONT_MEDIUM.render(f'High Score: {self.high_score}', True, Colors.TEXT_COLOR)
        restart_text = FONT_SMALL.render('Press SPACE to restart', True, Colors.TEXT_COLOR)
        quit_text = FONT_SMALL.render('Press ESC to quit', True, Colors.TEXT_COLOR)
        
        # Position text on panel
        panel.blit(game_over_text, (panel_width // 2 - game_over_text.get_width() // 2, 30))
        panel.blit(score_text, (panel_width // 2 - score_text.get_width() // 2, 120))
        panel.blit(high_score_text, (panel_width // 2 - high_score_text.get_width() // 2, 180))
        panel.blit(restart_text, (panel_width // 2 - restart_text.get_width() // 2, 250))
        panel.blit(quit_text, (panel_width // 2 - quit_text.get_width() // 2, 290))
        
        # Draw panel centered
        panel_x = (SCREEN_WIDTH - panel_width) // 2
        panel_y = (SCREEN_HEIGHT - panel_height) // 2
        self.screen.blit(panel, (panel_x, panel_y))
    
    def draw(self):
        """Draw everything"""
        # Background
        draw_gradient_background(self.screen)
        draw_grid(self.screen)
        
        # Game objects
        self.food.draw(self.screen)
        self.snake.draw(self.screen)
        
        # UI
        self.draw_ui()
        
        # Game over screen
        if self.game_over:
            self.draw_game_over()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(GAME_SPEED)

if __name__ == '__main__':
    game = SnakeGame()
    game.run()