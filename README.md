# Super-Mario-Bros-v0
A Reinforcement Learning agent designed to learn and complete the OpenAI Gym Super Mario Bros environment. These environments allow 3 attempts (lives) to make it through the 32 stages in the game. The environments only send reward-able game-play frames to agents.

## OpenAI Gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like pong or pinball. Gym is an open source interface to reinforcement learning tasks.

## Reinforcement learning algorithms
- Double Deep Q-learning (off-policy, model-free)

## Demo video
https://www.youtube.com/watch?v=O2QaSh4tNVw

## Requirements
- [Python 3.6 or 3.7](https://www.python.org/downloads/release/python-360/)
- [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- [cuDNN v7.6.5](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- [Pipenv](https://pypi.org/project/pipenv/)

## How to install the packages
You can install the required Python packages using the following command:
- `pipenv sync`

## How to run it
You can run the script using the following command: 
- `pipenv run python super_mario_bros_v0_ddqn.py`

# Note for developers
- you are responsible to update the default values of the hyperparameters
- research will continue once I get better hardware
