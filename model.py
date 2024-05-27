# Here we create our Q-Deep Learning model
# We will be using torch as our python RL library. See https://pytorch.org/docs/stable/index.html for 

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #torch.nn.function creates stateless (values are not saved between executions) functions that 
# Simply apply the operations such as convolution or relu without retaining any weights
import os 


class Linear_QNet(nn.Module): # Create a class for the linear Q network. This class inherets the nn.Module class from pytorch

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__() # super() gives access to the parent class, we are running the initialise function in the parent class nn.Module
        self.linear1 = nn.Linear(input_size, hidden_size) # Creates the first and input layer in the QNet
        self.linear2 = nn.Linear(hidden_size, output_size) # Creates the second and output layer in the QNet

    def forward(self, x): # Creates feed forward
        x = F.relu(self.linear1(x)) # Applies the relu function onto the first layer
        x = self.linear2(x) # Applies the output layer onto the first layer + relu activation
        return x

    def save(self, file_name = "model.pth"): # Creates function for saving the model
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path): # If a folder of ./model does not already exist in the project directory
            os.makedirs(model_folder_path) # Makes folder named model
        file_path = os.path.join(model_folder_path, file_name) # Sets the file path to a combined path of the folder + the file name
        torch.save(self.state_dict(), file_path) # Saves the model into the path


class QTrainer: # Create the Q Learning training model

    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr # Learning Rate is the pace in which the model updates and learns the values of a paramter estimate
        self.gamma = gamma # Gamma is the discount factor. It quantifies how much importance we give for future rewards. 
        self.model = model # The model
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr) 
        # self.model.paramaters() Gives the optimization model Adam the different paramaters to optimize (It's a iterable)
        # lr is learning rate
        self.criterion = nn.MSELoss()

    def train_step(self, current_state, action, reward, next_state, game_over):
        # Note before code: Tensors are multidimentional matrix that contain elements of a single time
        # We use tensors to store and operate on everything here
        # They are basically numpy arrays but can tackle heavy matrix manipulation for Deep learning purposes
        
        current_state = torch.tensor(current_state, dtype = torch.float) # Makes current state a tensor with a type of float
        action = torch.tensor(action, dtype = torch.long) # Rince and repeat
        reward = torch.tensor(reward, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)

        if len(current_state.shape) == 1: # IF the current tensor is a single dimension, this if condition unsqueezes it into a 2D array
            current_state = torch.unsqueeze(current_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state ,0)
            game_over = (game_over, ) # Makes game_over into a tuple, thereby making it unmutable

        prediction = self.model(current_state) # Gets the predicated value with the current state from the model (QNet)
        target = prediction.clone() # 
        # Note: game_over has a new value every iteration of the game, storing whether or not the game was over that turn
        for idx in range(len(game_over)): # Cycles through all turns 
            q_new = reward[idx] # Use the index above to check the respective reward values 
            if not game_over[idx]: # If the game was not over 
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) 
                # Sets q to the current reward + gamma value * max value of model output
            target[idx][torch.argmax(action[idx]).item()] = q_new # If the game is over

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()


