import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=0, fc4_units=0, dueling = False, fc_duel1 = 0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer (optional)
            fc4_units (int): Number of nodes in fourth hidden layer (optional, requires 3rd layer provided)
            dueling (bool): If True implement a Dueling DQN (1st hidden layer after the fork is of fc_duel1 size). Only works if 4 hidden layers are used (fc3_units and fc4_units are non nil)
            fc_duel1 (int): Number of nodes in the 1st layer of the dueling network (the branch for the state values)
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        if fc3_units > 0:
            self.fc3 = nn.Linear(fc2_units, fc3_units)
            if fc4_units > 0:
                self.fc4 = nn.Linear(fc3_units, fc4_units)
                if dueling:
                    self.fc5 = nn.Linear(fc4_units+1, action_size) # add 1 input to last layer
                else:
                    self.fc5 = nn.Linear(fc4_units, action_size)
                self.layers_number = 5
            else:
                dueling = False  # dueling network only implemented with 4 hidden layers
                self.fc4 = nn.Linear(fc3_units, action_size)
                self.layers_number = 4
        else:
            dueling = False # dueling network only implemented with 4 hidden layers
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.layers_number = 3
        if dueling: 
            self.fc_duel1 = nn.Linear(fc2_units, fc_duel1)
            self.fc_duel2 = nn.Linear(fc_duel1, 1)
        
        self.duelingDQN = dueling
        print('QNetwork ({} layers, dueling:{}) initialized with states space size {} and {} actions'.format(self.layers_number, self.duelingDQN, state_size, action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = None
        if self.layers_number > 3:
            q = F.relu(self.fc3(x))
            if self.duelingDQN:
                v = F.relu(self.fc_duel1(x))
            if self.layers_number > 4:
                q = F.relu(self.fc4(q))
                if self.duelingDQN:
                    v = F.relu(self.fc_duel2(v))
                    concatenated = torch.cat((q, v), 1)
                    x = self.fc5(concatenated)
                else:
                    x = self.fc5(q)
            else:
                x = self.fc4(q)
        else:
            x = self.fc3(x)
        return x
