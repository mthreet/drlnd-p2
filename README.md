# Project 1: Navigation

## Introduction
This project involved training robotic arms to move in a continuous action-space and state-space. The goal of the arms is to stay inside of a green sphere that is moving around them. The environment that was used had 20 parallel instances of the arms for faster training.

A reward of +0.1 is given for all time steps that an arm stays inside its green sphere. A reward of 0 is given for all time steps that an arm is outside its green sphere. Each arm receives an observation with 33 variables corresponding to its position, rotation, velocity, and angular velocity. Each arm can apply a torque to its two joints, resulting in an action space that is a 4-vector, corresponding to applicable torques. The torques can take on any continuous value between -1 and +1.

This problem was solved by using the Deep Deterministic Policy Gradient (DDPG) algorithm. DDPG is tailored to work well with continous action spaces. DDPG is an acotr-critic method, so it is not as susceptible to a bias-variance tradeoff as a pure policy-based method or value-based method. DDPG uses an actor network to deterministically output the optimal policy for each state, and a critic network to learn the action-value function for each state and optimal action. This allows the acotr and critic network to jointly optimize their weights, which helps avoid a bias-variance tradeoff.

## Installation
Follow the instuctions at [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies) (specifically the "Dependencies" section) for information on installing the environment. **Step 4 about the iPython kernel can be ignored**. This will require Anaconda for Python 3.6 or higher.

After the environment is ready, clone this repository:
```
git clone https://github.com/mthreet/drlnd-p2
```

## Running the code
To run a pretrained model (that received an average score of +13.0 over 100 episodes), simply run [eval.py](eval.py):
```
python eval.py
```

To train a model with [train.py](train.py), simply run:
```
python train.py
```
**Note that this will overwrite the checkpoint unless the save name is changed on line 58 of [train.py](train.py). Line 21 of [eval.py](eval.py) must also then be changed to the new corresponding name.**