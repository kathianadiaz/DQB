# Deep Q-learning Box (DQB)

## What is DQB?
  DQB’s main focus lies in the encapsulation of complex or convoluted code required to set up a working Q-learning agent while providing a set of environments to train and test the agent. This will allow Reinforcement Learning (RL) begginers to create, train and refine a RL agent without the overwhelming mathematics that come with it. The user will be able to modify the agent parameters and analize how they affect the training process and consequently the accuracy of the model. 

Accept the challenge and learn Reinforcement Learning with DQB!

## Setup

To get started, clone the repository to your computer. 
Make sure you have Python 3 and the PLY parsing tool installed. 
Afterwards, you need to install the external libraries required by using the command: 
pip install –r requirements.txt

Now you have everything to write in DQB!

*Ubuntu or Mac only. 

Instructions: 

First, to get started, use the command: python3 DQB.py 

Write the DQB source code on a .txt file called DQB_script. 

Run the DQB_blackbox which initializes, compiles, and trains the agent. 

Observe the agent's training process! You have the option to display the environment as it trains and the model's status episode by episode. 

## Language Features

### Creation of a simplified DQB agent:

With premade parameters for each agent, one would be able to change its internal properties. 
One would alter the agent’s parameters to see the effect it had on its ability to perform within an environment.

### Training and Testing:

Simple environments held within the Gym library will provide a way to test and train a model.
Actual state of the model while training will provide an insight of how the model is learning.

### DQN Architecture:
First, the user writes the DQN source code on a .txt file called DQB_script. The user then runs the DQB_blackbox which reads the source file, tokenizes it, parses it and generates the intermediate Python code. The intermediate code creates, initializes, compiles, and trains the agent. Finally, the user is able to observe the agent's training process having the option to display the environment as it trains and the model's status episode by episode. 

### Tools and Libraries
- TensorFlow: is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.
- Keras: Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.   
- Gym: Gym is a toolkit for developing and comparing reinforcement learning algorithms. It makes no assumptions about the structure of your agent, and is compatible with any numerical computation library, such as TensorFlow or Theano.
- PLY: PLY is an implementation of lex and yacc parsing tools for Python
-Others: h5py, numpy, skimage, atari-py

## Video example


## Final report 

To read the final report, find it in the repository by "Phase 3 - DQB.pdf" or use this link: https://github.com/kathianadiaz/DQB/blob/master/Phase%203%20-%20DQB.pdf 

## Developers

Guillermo Betancourt
Fabiola Badillo 
Kathiana Diaz

