# Here we setup and compile our agent

import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from snakeGame import SnakeGameAI, Direction, Point
from helper import plot


# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005


class Agent:

    def __init__(self) -> None:
        self.n_games = 0 # No. of games
        self.epsilon = 0 # Control the randomness
        self.gamma = 0.9 # Discount Rate Should be less than 1
        self.memory = deque(maxlen=MAX_MEMORY) # We use deque to store the list, deque is a dynamic array and provides random access
        # MAX_MEMORY is basically thea mount of variables that can be stored in the deque
        self.model = Linear_QNet(11, 256, 3) # Using our QNet model in model.py, we make a QNet of 11 input size, 256 hidden size, and 3 output size
        self.trainer = QTrainer(self.model, LR, self.gamma) # Using our QTrainer in model.py, we use the QNet model in it

    def get_state(self, game):
        # Get the position of a multitiude of objects within the game such as food positions, danger positions & move directions. 
        # Feed into numpy array
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food Location
            game.food.x < game.head.x,  # Food is to the left of the snakes current position
            game.food.x > game.head.x, # Food is to the right of the snakes current position
            game.food.y < game.head.y, # Food is to the top of the snakes current position
            game.food.y > game.head.y # Food is to the bottom of the snakes current position
        ]
        return np.array(state, dtype=int) # dtype used to convert bools to int
    
    def remember(self, state, action, reward, next_state, game_over):
        # Use this for memory
        self.memory.append((state, action, reward, next_state, game_over)) # Appends all values into the memory


    # Quick Note: We use both long memory and short memory training. Short memory training is basically consecutive training
    # During a game iteration
    # Long memory training or expierence relay. In a nutshell expierence relay takes a random sample from memory and trains on that.
    # The reason we use expierence relay is because it brakes correlation between consecutive samples. These samples are 
    # Highly correlated and will cause inefficient learning
    def train_long_memory(self):
        # Only do random sampling if the memory was above sample size (Otherwise too little training data)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Takes a batch size amoutns of random sample from memory
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)


    def get_action(self, state):
        # Get random move: Tradeoff between exploration and expoitations
        self.epsilon = 80 - self.n_games # Get Epsilon value
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get the current state
        current_state = agent.get_state(game)
        # Get Move
        final_move = agent.get_action(current_state)

        # Play the move and get new state
        reward, game_over, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # Train Short Memory
        agent.train_short_memory(current_state, final_move, reward, new_state, game_over)

        # Remember
        agent.remember(current_state, final_move, reward, new_state, game_over)

        if game_over:
            # Train Long Memory or Experience Memorty or Replay and plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")
            # Plot the graphs
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if(__name__ == "__main__"):
    train()
