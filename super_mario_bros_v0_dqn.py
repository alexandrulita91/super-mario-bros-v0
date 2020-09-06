"""
Super-Mario-Bros-v0 -- Deep Q-learning with Experience Replay
"""

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import wrap_nes
import cv2
import numpy as np

env = wrap_nes("SuperMarioBros-1-1-v0", SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())
    print(env.observation_space.low.shape)
    # print(env.observation_space.high)
    env.render()
    cv2.imshow('image', np.array(observation))
    # print(env.observation_space.low)
    # print(np.array(observation)[0])
    # cv2.imwrite("hello.jpg", observation)
    # print(observation)
    cv2.waitKey(0)

env.close()