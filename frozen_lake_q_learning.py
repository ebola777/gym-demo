'''OpenAI Gym FrozenLake-v0 using Q-learning
'''
import copy

import numpy as np

from gym_wrapper import GymWrapper
from rl import (AcceptanceEvaluator, DiscreteEnvioronment, EpisodicTrainer,
                get_state_set, get_action_set, TabularQLearning,
                TabularQLearningParameter, Monitor)


class AppMonitor(Monitor):
  learner = None
  episode_index = 0
  time_step = 0
  done = False
  last_avg_reward = 0.0

  def __init__(self, decay_factor_non_greedy_factor,
               decay_factor_learning_rate):
    # Save the parameters
    self.decay_factor_non_greedy_factor = decay_factor_non_greedy_factor
    self.decay_factor_learning_rate = decay_factor_learning_rate
    # Create an acceptance evaluator
    avg_reward_thresh = 0.78 + 0.01
    time_step_size = 100
    self.acceptance_evaluator = AcceptanceEvaluator(
        avg_reward_thresh, time_step_size)

  def notify_training_start(self, learner, init_result):
    del init_result
    self.learner = learner

  def notify_episode_start(self, episode_index, init_result):
    del init_result
    self.episode_index = episode_index
    # Decay
    param = self.learner.param
    param.non_greedy_prob *= (
        1.0 - self.decay_factor_non_greedy_factor)
    param.learning_rate *= (1.0 - self.decay_factor_learning_rate)
    self.last_avg_reward = self.acceptance_evaluator.avg_reward

  def notify_time_step_start(self, time_step, init_result):
    del init_result
    self.time_step = time_step

  def notify_env_response(self, done, info):
    del info
    self.done = done

  def notify_learning_result(self, observation, learning_result):
    del learning_result
    if self.done:
      # Add the reward to acceptance evaluator
      self.acceptance_evaluator.add_reward(observation.reward)
      if observation.reward > 0:
        avg_reward = self.acceptance_evaluator.get_avg_reward()
        print('Average reward: {}'.format(avg_reward))
        print('Episode index: {}'.format(self.episode_index))
        print('Episode finished after {} timesteps'.format(self.time_step + 1))

  def get_observation(self, observation):
    modified_observation = copy.copy(observation)
    if self.done:
      if modified_observation.reward > 0.0:
        modified_observation.reward = 1.0
      else:
        modified_observation.reward = 0.0
    return modified_observation

  def check_episode_stop(self):
    return self.acceptance_evaluator.is_accepted()


def init_gym_wrapper():
  config_file = 'config/gym_config.yaml'
  env_name = 'FrozenLake-v0'
  gym_wrapper = GymWrapper(config_file)
  gym_wrapper.create_env(env_name)
  return gym_wrapper


def init_rl_env(gym_env):
  state_set = get_state_set(gym_env)
  terminal_state_set = [15]
  action_set = get_action_set(gym_env)
  rl_env = DiscreteEnvioronment(state_set, terminal_state_set, action_set)
  return rl_env


def init_q_learning(rl_env):
  non_greedy_prob = 0.5
  learning_rate = 1.0
  discount_factor = 0.999
  param = TabularQLearningParameter(rl_env, non_greedy_prob, learning_rate,
                                    discount_factor)
  tabular_q_learning = TabularQLearning(param)
  # Initialize Q table
  random_state = np.random.RandomState(0)
  tabular_q_learning.q_table = {}
  for state in rl_env.state_set:
    for action in rl_env.action_set:
      sa_pair = (state, action)
      if not state in rl_env.terminal_state_set:
        rand_val = random_state.rand()
        tabular_q_learning.q_table[sa_pair] = 0.5 + 1e-6 * rand_val
      else:
        tabular_q_learning.q_table[sa_pair] = 0.0
  return tabular_q_learning


def main():
  # Initialize Gym
  gym_wrapper = init_gym_wrapper()
  gym_env = gym_wrapper.env
  # Initialize RL
  rl_env = init_rl_env(gym_env)
  q_learning = init_q_learning(rl_env)
  # Create a app monitor
  decay_factor_non_greedy_factor = 0.005
  decay_factor_learning_rate = 0.001
  monitor = AppMonitor(decay_factor_non_greedy_factor,
                       decay_factor_learning_rate)
  # Create a episodic trainer
  episode_size = 10000
  time_step_size = 100
  trainer = EpisodicTrainer(episode_size, time_step_size)
  # Train
  trainer.run(gym_env, monitor, q_learning)
  # Close the Gym enviornment
  gym_env.close()
  # Upload the results if the performance is accepted
  if monitor.acceptance_evaluator.is_accepted():
    gym_wrapper.upload_result()
  # Finalize
  print('Learning ended after {} episodes'.format(monitor.episode_index + 1))
  print('Average reward: {}'.format(monitor.acceptance_evaluator.avg_reward))


if __name__ == '__main__':
  main()
