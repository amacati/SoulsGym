"""The ``soulsgym`` package is a collection of OpenAI gym environments for Dark Souls III.

It contains two main components. The first module :mod:`soulsgym.envs` includes a core environment
as well as an individual environment for each available boss fight. SoulsGym uses Dark Souls III as
the underlying engine that is modified at runtime by reading and writing into the game memory to
create the environments.

Note:
    The environments only cover one phase of the boss fights so far. For further explanations see
    :mod:`~.envs`.

We do however provide a demo environment. This environment is meant to test the agent on a single
episode of the full boss fight.

The second main module is the :mod:`soulsgym.core` module. It contains all necessary
functionalities for the environments to interact with the game. Unless you want to develop your own
environment for :mod:`~.envs` or are trying to contribute, this module can safely be ignored.

``soulsgym`` registers its environments with OpenAI's ``gym`` module on import. In order to use the
environments you follow the usual pattern of OpenAI's ``gym.make``. A list of all available
environments is available at ``soulsgym.available_envs``.
"""
import logging

from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)


def set_log_level(level: int):
    """Set log level for the soulsgym module.

    Args:
        level: Logger level for the module. Uses the standard library logging module levels.
    """
    logger.setLevel(level)


available_envs = ["SoulsGymIudex-v0", "SoulsGymIudexDemo-v0"]

# Register environments in OpenAI gym
register(id="SoulsGymIudex-v0",
         entry_point='soulsgym.envs.darksouls3.iudex_env:IudexEnv',
         max_episode_steps=3000,
         nondeterministic=True)
register(id="SoulsGymIudexDemo-v0",
         entry_point="soulsgym.envs.darksouls3.iudex_env:IudexEnvDemo",
         max_episode_steps=3000,
         nondeterministic=True)
