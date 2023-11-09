"""The SoulsGym environment for Vordt of the Boreal Valley.

The player and Vordt always start from their respective start poses at full HP/SP. The player starts
with the stats and weapons as configured in <TODO: add config>. We do not allow shield blocking or
two handing at this point, although this can easily be supported. Parrying is enabled.

Note:
    Phase 2 of the boss fight is available by setting the environment keyword argument ``phase``.
    See :mod:`~.envs` for details.
"""
from __future__ import annotations

import logging

from soulsgym.envs.soulsenv import SoulsEnv

logger = logging.getLogger(__name__)


class VordtEnv(SoulsEnv):
    """The SoulsGym environment for Vordt of the Boreal Valley."""

    def __init__(self, game_speed: float = 1., phase: int = 1):
        """Initialize the observation and action spaces.

        Args:
            game_speed: The speed of the game during :meth:`.SoulsEnv.step`. Defaults to 1.0.
            phase: The phase of the boss fight. Either 1 or 2 for Vordt. Defaults to 1.
        """
        super().__init__(game_speed=game_speed)

        self.phase = phase
