"""GameState data class for storing the internal game state."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GameState:
    """Represent a snapshot-state of the game."""

    phase: int = 1
    player_hp: int = 0
    player_max_hp: int = 0
    player_sp: int = 0
    player_max_sp: int = 0
    boss_hp: int = 0
    boss_max_hp: int = 0
    player_pos: np.ndarray = np.zeros(4, dtype=np.float32)
    boss_pos: np.ndarray = np.zeros(4, dtype=np.float32)
    player_animation: str = "NoAnimation"
    player_animation_count: int = 0
    boss_animation: str = "NoAnimation"
    boss_animation_count: int = 0
    locked_on: bool = False

    def copy(self) -> GameState:
        """Copy the object.

        Returns:
            A copy of itself.
        """
        return GameState(**self.__dict__)

    def __getitem__(self, name: str) -> Any:
        """Enable attribute access by key indexing.

        Args:
            name: Attribute name.

        Returns:
            The attribute value.
        """
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any):
        """Enable attribute assignment by key indexing.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        setattr(self, name, value)
