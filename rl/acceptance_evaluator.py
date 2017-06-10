'''OpenAI Gym acceptance evaluator
'''
from collections import deque


class AcceptanceEvaluator(object):
  total_reward = 0.0
  avg_reward = 0.0

  def __init__(self, avg_reward_thresh, time_step_size):
    # Save the parameters
    self.avg_reward_thresh = avg_reward_thresh
    self.time_step_size = time_step_size
    # Reset the reward
    self.reset()

  def reset(self):
    self.reward_list = deque(maxlen=self.time_step_size)
    self.total_reward = 0.0
    self.avg_reward = 0.0

  def add_reward(self, reward):
    # Subtract from the leftmost element if the queue is full
    if len(self.reward_list) >= self.reward_list.maxlen:
      self.total_reward -= self.reward_list[0]
    # Add the new reward to the total reward
    self.total_reward += reward
    self.reward_list.append(reward)
    # Calculate the average reward
    self.avg_reward = self.total_reward / len(self.reward_list)

  def get_avg_reward(self):
    return self.avg_reward

  def is_accepted(self):
    return self.avg_reward >= self.avg_reward_thresh
