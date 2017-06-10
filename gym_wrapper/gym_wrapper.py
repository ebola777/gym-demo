'''OpenAI Gym wrapper

References:
  - http://alanwsmith.com/capturing-python-log-output-in-a-variable
'''
import io
import logging
import re
import webbrowser

from collections import namedtuple
from gym import envs
from gym import wrappers
import gym
import yaml


class GymWrapper(object):
  config = None
  env = None

  def __init__(self, config_file):
    self.read_config_file(config_file)

  @staticmethod
  def print_all_env_ids():
    env_ids = [spec.id for spec in envs.registry.all()]
    sorted_env_ids = sorted(env_ids)
    for env_id in sorted_env_ids:
      print(env_id)

  def read_config_file(self, config_file):
    with open(config_file, 'r') as stream:
      obj = yaml.safe_load(stream)
    stream.close()
    self.config = namedtuple('GymConfig', obj.keys())(*obj.values())

  def create_env(self, env_name):
    self.env = gym.make(env_name)
    if self.config.record:
      self.env = wrappers.Monitor(
          self.env, directory=self.config.monitor_dir, force=True)
    return self.env

  def upload_result(self):
    if self.config.upload:
      # Get the Gym API logger
      logger = logging.getLogger('gym.scoreboard.api')
      # Setup the console handler with a StringIO object
      log_capture_string = io.StringIO()
      stream_handler = logging.StreamHandler(log_capture_string)
      stream_handler.setLevel(logging.DEBUG)
      # Add the console handler to the logger
      logger.addHandler(stream_handler)
      # Upload the result
      gym.upload(self.config.monitor_dir, api_key=self.config.api_key)
      # Pull the contents back into a string and close the stream
      log_contents = log_capture_string.getvalue()
      log_capture_string.close()
      # Parse the contents
      pattern = re.compile(r'You can find it at:\n\n\s+(?P<url>.+)\n\n')
      match = pattern.search(log_contents)
      evaluation_url = match.group('url')
      # Open the URL in the browser
      webbrowser.open_new_tab(evaluation_url)
    else:
      print('Config option "upload" disables the uploading')

  def print_env(self):
    print('Action space: {}'.format(self.env.action_space))
    print('Observation space: {}'.format(self.env.observation_space))


class GymConfig(object):
  api_key = None
  monitor_dir = None
  env_name = None
  record = False
  upload = False
