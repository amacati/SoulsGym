"""The ``env`` module contains the SoulsGym environment base class and all boss environments.

All ``soulsgym`` environments implement the abstract methods defined in the abstract base
environment :class:`~.soulsenv`. This base class contains the general gym logic and setup.

Each boss environment defines its own class by inheriting from :class:`~.soulsenv`. All environments
have to define the ``ENV_ID`` attribute and a matching config file in the config folder. Most of the
boss specific setup has to be done inside the individual environments.

Note:
    The game is used as 'engine' for the environments. It has to be running before any environments
    are created!

Once the fight starts, ``soulsgym`` keeps the player and the boss alive and at full health. HP
losses are tracked internally instead. After the gym has determined that either the player or the
boss has died it resets the poses and animations of the player and the boss. This is done to save
time on the reload of the game. As soon as the initial state is restored, the gym continues with a
new episode.  Alternatively, an episode ends after 3000 steps (5 minutes). Sometimes, the player can
get stuck in a weird position. The time limit ensures frequent resets to aleviate this problem.

Note:
    Since we always top off the boss' HP it never enters its second phase. We support training on
    different phases by specifying the phase on environment initialization. This phase cannot be
    changed during runtime. We recommend training one agent for each phase instead.

Warning:
    Do not attempt to launch more than one environment at once! There can only be one instance of
    Dark Souls III. Multiple ``soulsgym`` environments would conflict with each other by
    reading and writing to the same game instance!

During training ``soulsgym`` uses the Windows API for Python to control the player with keystrokes.
This has two consequences: First, the user should refrain from pressing any buttons during
training as this would result in player actions that are not controlled by the gym. Second, the game
window has to remain focussed at all times for the game to register the keystrokes.

Warning:
    Do **not** interact with your game in any way during training.
"""

import logging

import pymem

# Disable pymem logging
pymem.logger.setLevel(logging.WARNING)
