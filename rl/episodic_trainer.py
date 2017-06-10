'''Episodic RL trainer
'''
from rl.environment import Observation


class EpisodicTrainer(object):
  def __init__(self, episode_size, time_step_size):
    self.episode_size = episode_size
    self.time_step_size = time_step_size

  def run(self, gym_env, monitor, learner):
    # Initialize the learner in the training session
    init_result = learner.init_training()
    # Notify the monitor about the training session
    monitor.notify_training_start(learner, init_result)
    for episode_index in range(self.episode_size):
      # Initialize the Gym environment
      state = gym_env.reset()
      # Initialize the learner in the current episode session
      init_result = learner.init_episode(episode_index, state)
      # Notify the monitor about the starting episode
      monitor.notify_episode_start(episode_index, init_result)
      for time_step in range(self.time_step_size):
        # Initialize the learner in the current time step session
        init_result = learner.init_time_step(time_step, state)
        # Notify the monitor about the starting time step
        monitor.notify_time_step_start(time_step, init_result)
        # Choose the action
        action = learner.choose_action(state)
        # Do the action in the Gym environment
        (next_state, reward, done, info) = gym_env.step(action)
        # Notify the monitor about the environment response
        monitor.notify_env_response(done, info)
        # Build the observation
        observation = Observation(state, action, reward, next_state)
        # Get the observation from the monitor
        modified_observation = monitor.get_observation(observation)
        # Learn
        learning_result = learner.learn(modified_observation)
        # Notify the monitor about the learning result
        monitor.notify_learning_result(observation, learning_result)
        # Check whether the Gym enviornment has terminated
        if done:
          break
        # Check wether the monitor indicates to stop the time step session
        if monitor.check_time_step_stop():
          break
        # Transition the state
        state = next_state
      # Check wether the monitor indicates to stop the time step session
      if monitor.check_episode_stop():
        break
