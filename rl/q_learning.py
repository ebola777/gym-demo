'''Q-learning utility
'''
import numpy as np

from rl.learner import Learner


class TabularQLearning(Learner):
  q_table = {}

  def __init__(self, param):
    # Save the parameters
    self.param = param
    # Make the NumPy random generator predictable
    self.rand = np.random.RandomState(0)

  def choose_action(self, state):
    (env, non_greedy_prob, _, _) = self.param.get()
    rand_value = self.rand.rand()
    if rand_value > non_greedy_prob:
      state_q_value_set = [
          self.q_table[(state, action)] for action in env.action_set]
      max_index = np.argmax(state_q_value_set)
      return env.action_set[max_index]
    else:
      action_size = len(env.action_set)
      rand_index = np.floor(action_size * self.rand.rand())
      rand_index = rand_index.astype(int)
      return env.action_set[rand_index]

  def learn(self, observation):
    (env, _, learning_rate, discount_factor) = self.param.get()
    (state, action, reward, next_state) = observation.get()
    sa_pair = (state, action)
    sa_old_value = self.q_table[sa_pair]
    next_state_q_value_set = [
        self.q_table[(next_state, action)] for action in env.action_set]
    td_error = reward + \
        (discount_factor * np.max(next_state_q_value_set)) - sa_old_value
    td_error = np.asscalar(td_error)
    self.q_table[sa_pair] += learning_rate * td_error

  def print_q_table(self):
    env = self.param.env
    for state in env.state_set:
      for action in env.action_set:
        sa_pair = (state, action)
        q_value = self.q_table[sa_pair]
        print('({}, {}): {}'.format(state, action, q_value))


class TabularQLearningParameter(object):
  def __init__(self, env, non_greedy_prob=0.5, learning_rate=0.5,
               discount_factor=1.0):
    self.env = env
    self.non_greedy_prob = non_greedy_prob
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor

  def get(self):
    return (self.env, self.non_greedy_prob, self.learning_rate,
            self.discount_factor)
