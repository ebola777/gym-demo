'''The converter between RL raw data and OpenAI Gym
'''
import gym
import numpy as np


def get_state_set(gym_env):
  return _gym_space_to_set(gym_env.observation_space)


def get_action_set(gym_env):
  return _gym_space_to_set(gym_env.action_space)


def _gym_space_to_set(space):
  if isinstance(space, gym.spaces.Discrete):
    return np.arange(space.n).tolist()
  else:
    raise ValueError(
        'Gym environment class {} cannot be converted to discrete set'
        .format(type(space).__name__))
