# Copyright (c) 2023 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import numpy as np

import alf
from alf.environments import suite_gym
from vonet.movi import MOVi


@alf.configurable
def load(game,
         env_id=None,
         env_args=dict(),
         discount=1.0,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         max_episode_steps=0):
    """Loads the specified simple game and wraps it.
    Args:
        game (str): name for the environment to load. The game should have been
            defined in the sub-directory ``./simple/``.
        env_args (dict): extra args for creating the game.
        discount (float): discount to use for the environment.
        gym_env_wrappers (list): list of gym env wrappers.
        alf_env_wrappers (list): list of ALF env wrappers.
        max_episode_steps (int): max number of steps for an episode.

    Returns:
        An AlfEnvironment instance.
    """
    env = MOVi(**env_args)
    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        auto_reset=True)
