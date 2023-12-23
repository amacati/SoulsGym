"""The ``soulsgym`` package is a collection of Gymnasium environments for Dark Souls III.

Its high-level objective is to provide Souls games boss fights as reinforcement learning
environments. These environments are located in the :mod:`soulsgym.envs` module. It is structured
into a core environment that is subclassed into an individual environment for each available boss
fight. SoulsGym uses the Souls games, e.g. Dark Souls III or Elden Ring, as the underlying engine
that is modified at runtime by reading and writing into the game memory to create the environments.

Note:
    The environments only cover one phase of the boss fights so far. For further explanations see
    :mod:`~.envs`.

We also provide a demo environment. This environment is meant to test the agent on a single episode
of the full boss fight.

The Souls games have no API for developers to modify the game. Therefore, ``soulsgym`` has to hack
the games instead. The core hacking and interactions tools are located in the :mod:`soulsgym.core`
module. Unless you want to develop your own environment for :mod:`~.envs` or are trying to
contribute, this module can safely be ignored.

As an intermediate layer between the core module and the environments, the :mod:`soulsgym.games`
module provides a game class for each of the Souls games that is used to seamlessly interact with
the games as if they were regular Python objects. This immensely simplifies the development of new
environments.

``soulsgym`` registers its environments with the ``gymnasium`` module on import. In order to use the
environments you follow the usual pattern of ``gymnasium.make``. A list of all available
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

# Register environments in gymnasium
register(id="SoulsGymIudex-v0",
         entry_point='soulsgym.envs.darksouls3.iudex:IudexEnv',
         max_episode_steps=3000,
         nondeterministic=True)
register(id="SoulsGymIudexImg-v0",
         entry_point='soulsgym.envs.darksouls3.iudex:IudexImgEnv',
         max_episode_steps=3000,
         nondeterministic=True)
register(id="SoulsGymIudexDemo-v0",
         entry_point="soulsgym.envs.darksouls3.iudex:IudexEnvDemo",
         max_episode_steps=3000,
         nondeterministic=True)
register(id="SoulsGymVordt-v0",
         entry_point='soulsgym.envs.darksouls3.vordt:VordtEnv',
         max_episode_steps=3000,
         nondeterministic=True)