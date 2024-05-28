# In this file we create the snake game enviroment

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import globalVariables
from datetime import datetime

# Declare Constants
#rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 0, 200)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80


pygame.init()
font = pygame.font.Font('Times New Roman.ttf', 25)


class Direction(Enum): # We create an enumerator class for all directions
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y') # A named tuple is a basically a lightweight class. It's similar to dictionaries in the sense
# That it contains keys that are hashed to a specific value. However unlike dictionaries it supports key-value and iteration.
# We use a tuple here to store mulitple hashed immutable objects

# In this case point is a way we use to store the position of the snake and food.

class SnakeGameAI:
    def __init__(self, w = 640, h = 480) -> None: 
        # Use init to setup the training env and display (640 x 480)
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # Initialise the enviorment & game states & game objects
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ] # By Default the snake is size 3
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()
    
    def _place_food(self):
        random.seed(datetime.now().timestamp())
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE # Takes a random position 
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake: # If the position is already in the snake, run again
            self._place_food()

    def play_step(self, action):
        # Compiles all operations in the game here

        self.frame_iteration += 1
        # 1. Get user input
        for event in pygame.event.get():  # Input to quit
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if event.type == pygame.KEYDOWN:
            #     if event.type == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.type == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.type == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.type == pygame.K_DOWN:
            #         self.direction = Direction.DOWN 
            # If for some reason you want to play the game manually lol

        # 2. Move (Use move definition)
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if the game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake): # If there is a collision or snake name is constant (broken prob)
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check for food eaten, then move or place new fruit
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and tick game
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. If Game over, return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt = None):
        # Checks for collision, useful for game
        if pt is None:
            pt = self.head # Point is the head of the snake
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y < 0 or pt.y > self.h - BLOCK_SIZE: # If snake hits side of map
            return True
        if pt in self.snake[1:]: # If snake hits itself 
            return True
        return False
    
    def _update_ui(self):
        # Based on the game proccessed above, updates the UI shown
        self.display.fill(BLACK)
        for pt in self.snake: # Draws snake
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)) # Inside of snake is dark blue
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12)) # Outside (Outline) is lighter blue
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)) # Food is red

        scoreText = font.render(f"Score: {str(self.score)}", True, WHITE) # Display score top left
        self.display.blit(scoreText, [5, 50])
        recordText = font.render(f"Record: {str(globalVariables.recordGlobal)}", True, WHITE) # Display record 
        self.display.blit(recordText, [5, 25])
        gamesText = font.render(f"Game No.{str(globalVariables.gamesGlobal)}", True, WHITE) # Display the No. of Games
        self.display.blit(gamesText, [5, 0])

        seconds = int((pygame.time.get_ticks()/1000))
        minutes = int(seconds/60)
        timeText = font.render(f"Time Elapsed: {str(minutes)}m, {str(seconds%60)}s", True, WHITE)
        self.display.blit(timeText, [400, 0])
        pygame.display.flip()

    def _move(self, action):
        # All movement operation and computations
        #[straight, right, left]
        # Note that the model sees the game from the "perspective" of the model, we proccess movement in that way
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] # Creates a array of all directions in clockwise
        idx = clock_wise.index(self.direction) # Based on current direction, get the direction the snake is facing

        # Note for directional computations： "technically" a clockwise right turn is simply a +1 iterator. We use mod here to prevent
        # Out of bounds that will occure when you turn right from up direction
        if np.array_equal(action, [1, 0, 0]): # If the action chosen is forward
            new_dir = clock_wise[idx] # New direction is the same as the current
        elif np.array_equal(action, [0, 1, 0]): # If action chosen is right
            next_idx = (idx+1) % 4 
            new_dir = clock_wise[next_idx] 
        else: # If action chosen is left （We use else because the model will return a direction）
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] 
        self.direction = new_dir
        
        # Changes head position based on movement
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
    