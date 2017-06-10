'''RL learner interface
'''
from abc import ABCMeta, abstractmethod


class Learner(metaclass=ABCMeta):
  def init_training(self):
    pass

  def init_episode(self, episode_index, state):
    pass

  def init_time_step(self, time_step, state):
    pass

  @abstractmethod
  def choose_action(self, state):
    raise NotImplementedError()

  @abstractmethod
  def learn(self, observation):
    raise NotImplementedError()
