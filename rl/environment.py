'''RL environment
'''


class DiscreteEnvioronment(object):
  def __init__(self, state_set, terminal_state_set, action_set):
    self.state_set = state_set
    self.terminal_state_set = terminal_state_set
    self.action_set = action_set

  def get(self):
    return (self.state_set, self.terminal_state_set, self.action_set)


class Observation(object):
  def __init__(self, state, action, reward, next_state):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state

  def get(self):
    return (self.state, self.action, self.reward, self.next_state)
