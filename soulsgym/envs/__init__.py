"""The ``env`` module contains the SoulsGym environment base class and all boss environments.

All ``soulsgym`` environments implement the abstract methods defined in the abstract base
environment :class:`~.soulsenv`. This base class contains the general gym logic and setup.

Each boss environment defines its own class by inheriting from :class:`~.soulsenv`. All environments
have to define the ``ENV_ID`` attribute and a matching config file in the config folder. Most of the
boss specific setup has to be done inside the individual environments.

Once the fight starts, soulsgym keeps the player and the boss alive and at full health. HP losses
are tracked internally instead. After the gym has determined that either the player or the boss has
died it resets the poses and animations of the player and the boss. We do this to save time on the
reload of the game. Once the initial state is restored, the gym continues with a new episode.

Note:
    Since we always top off the boss' HP it never enters its second phase. We could allow for it to
    drop in the game as well, but transformation into the second phase is irreversible so far. Until
    we find the flags inside the game memory responsible for the phase change we will stick to the
    first phase only.
"""
import logging

import pymem

# Disable pymem logging
pymem.logger.setLevel(logging.WARNING)
