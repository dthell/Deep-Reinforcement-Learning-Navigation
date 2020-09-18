[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, the aim is to train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


### The code architecture

3 files are used:
- model.py: contains the Deep Neural Network code (architecture and forward propagation)
- DeepQNetwork_agent.py: contains the code of the agent using 2 DQN for the reinforcement learning task
- Navigation.ipynb: is the Jupyter notebook used to instanciate the environment, the agent and run its training or actions

The execution is done through the 3rd file, the Jupyter notebook. In this Jupyter notebook file, the sections 1 to 3 are largely inspired from Udacity Navigation Project from the Deep Reinforcement Learning Nanodegree.


### Dependencies

The code uses the following dependencies:
- Unity ML-Agents (https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
- Numpy (http://www.numpy.org/ , numpy version 1.19.1)
- collections from which deque and namedtuple are imported
- PyTorch (torch version 0.4.0)
- model from which QNetwork is imported

The versions are for indication only as the one used to develop the code but the code may run with other versions.


### Installation

1. The environment can be downloaded through one of the link below, please select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. Then create a new environment with Python. For a detailed view on installing this environment the other dependancies, please refer to the dependencies section of the DRLND Git Hub: [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)


### The Model

Various deep NN have been tested for this problem and the best and simplest design among the similar results for this environment is the following:

4 hidden layers NN, all fully-connected, with sizes:
- FC layer 1: 64 nodes
- FC layer 2: 128 nodes
- FC layer 3: 128 nodes
- FC layer 4: 64 nodes

A ReLU activation fonction is used on each layer except the final (5th)

The training was done with the hyperparameters:
- Epsilon: starts at 1 and is multiplied by 0.99 at every episode with a minimum value of 0.01. Learning was sped up compared to a decay rate of 0.995 with the same type of score after sufficient training.
- Learning Rate: 1e-3. Learning was a bit faster for all tests done than with a learning rate of 5e-4
- Tau: 1e-3 for the update of the target Qnetwork weights. This hyper parameter was not tested for optimization
- Updates of the target QNetwork was done every 4 steps. Moving it to a higher number such as 10 is slowing down significantly the learning for no improved performance in the long run
- Gamma: 0.99. The discount factor was not tested for optimization

Trained parameters are provided for this QNetwork (see testing the trained agent section below).

The DQN used is a vanilla DQN. A dueling network and the prioritized replay do not seem to bring much to this problem but would slow down the learning, especially the prioritized DQN. However a Dueling DQN with trained parameters is also provided with very similar results to the vanilla DQN.


### Running the code

Please run the file [Navigation.ipynb](Navigation.ipynb) to load the environment, investigate it, test a random agent, train an agent and run previously trained agents.

1. The environment and the required dependencies are loaded from the [Start the Environment](Navigation.ipynb#Start-the-Environment) section.

 This section then launches the environment through the cell: 
```python
env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")
```
 It opens a new window where the progress of the agent can be monitored.

 Then final cell in this section checks for the first brain available in the environment, a brain controls its associated agents, as the default to be controled from Python.

2. The second [section](Navigation.ipynb#Examine-the-State-and-Action-Space) looks at the space and action spaces.

 Running the cell code here will display information and data structures about agents, actions and spaces.

3. The 3rd [section](Navigation.ipynb#Take-Random-Actions-in-the-Environment) is about testing a random action agent.

 By running the cell you will see in the separate window an untrained agent taking random steps. At the end of 300 steps, the score will be displayed.

4. The 4th [section](Navigation.ipynb#Training-the-agent) is about training an agent.

 It first sets the environment into training mode than then instantiate an agent to be trained.

 You can change the QNetwork used by the agent you want to train by changing the parameters of the following line:
```python
agent = Agent(state_size=len(env_info.vector_observations[0]), action_size=brain.vector_action_space_size, seed=0, fc1=64, fc2=128, fc3=128, fc4=64, dueling = True, fcduel = 32)
```

 If the agent succeeds in reaching out a score of +13 in less than 800 episodes, the trained weights will be written in a file named CheckpointN.pth with N being the trial number as passed in the paramters of function dqn()

 The scores are also output and a graph showing the evolution of the score with the episode can be plotted by running the last cell in this section.

5. The last [section](Navigation.ipynb#Testing-the-trained-agent) is about watching how trained agents are performing in the environment.

 The first cell runs a trained agent using vanilla DQN.

 The second cell runs another trained agent using a dueling DQN.

 Finally the last cell block closes the environment. This closes the separate window.


### How to test other DQN architectures

Other type of architectures can be tested by varying the input parameters to the Agent object at the begining of [section 4](Navigation.ipynb#Training-the-agent) which calls 2 QNetwork objects with the following constraints:
- minimum number of hidden layers is 2 : fc1 and fc2 of the class Agent must be non zero
- the size of the 4th hidden layer (fc4) is considered only if there is a 3rd hidden layer (so if fc3 is non zero also)
- dueling is used to instantiate a dueling DQN. This parameter is considered only with 4 hidden layers: 2 shared layers and 2 layers in the dueling phase
- if dueling is True and a non zero size is provided on the 4 hidden layers parameters (fc1 to fc4), a Dueling DQN is implemented and the 1st layer of the dueling phase is of size fcduel in the Agent object

Prioritized replay can also be tested by setting the parameter alpha of the Agent object to a value between 0 and 1 (0 being a uniform random sampling of the experience replay, 1 being a fully prioritized random sampling of the experience replay using the computed TD errors as weights). The beta parameter is fixed in the DeepQNetwork_agent.py file but was not optimized (see report.pdf).


### Testing the trained agents

A set of trained parameters is provided for the agent described in the Model section above and recalled here:
4 hidden layers vanilla DQN, all fully-connected with ReLU activation functions, with sizes:
- FC layer 1: 64 nodes
- FC layer 2: 128 nodes
- FC layer 3: 128 nodes
- FC layer 4: 64 nodes

The agent can be tested by running the cell after [Testing the trained agent (vanilla DQN)](Navigation.ipynb#Testing-the-trained-agent). The training of this agent reached a score of 13 over 100 consecutive episodes before reaching the 400th episode (average score of 13.15 over episodes 301 to 400).


As an alternative, a dueling DQN has been trained and can be tested by running the second cell block in this section right after "Testing the trained agent (dueling DQN)". The training of this agent reached a score of 13 over 100 consecutive episodes before reaching the 400th episode (average score of 13.3 over episodes 301 to 400).

This agent uses the following QNetwork architecture
- FC layer 1: 64 nodes
- FC layer 2: 128 nodes
- FC layer 3: 128 nodes    &    dueling FC layer 1: 128 node
- FC layer 4: 64 nodes     &    dueling FC layer 2: 1 node
The final layer takes 65 inputs (aka 64 + 1) and outputs 4 value as for the action space size
The activation fonction is a ReLU activation fonction for all layers except the final one. This final one is a plain linear fonction.


### Further details

A more detailled explanation of the project and its outcomes is provided in the report.pdf next to this README file.
