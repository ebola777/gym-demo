'''RL monitor
'''
from abc import ABCMeta


class Monitor(metaclass=ABCMeta):
  def notify_training_start(self, learner, init_result):
    pass

  def notify_episode_start(self, episode_index, init_result):
    pass

  def notify_time_step_start(self, time_step, init_result):
    pass

  def notify_env_response(self, done, info):
    pass

  def notify_learning_result(self, observation, learn_result):
    pass

  def get_observation(self, observation):
    return observation

  def check_time_step_stop(self):
    return False

  def check_episode_stop(self):
    return False
