"""The ``GameState`` is a ``dataclass`` that contains all information about the game state.

It is also the observation type returned by soulsgym steps and resets and used as the internal state
representation of the gym.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GameState:
    """Collect all game state information in a single data class."""

    phase: int = 1
    player_hp: int = 0
    player_max_hp: int = 0
    player_sp: int = 0
    player_max_sp: int = 0
    boss_hp: int = 0
    boss_max_hp: int = 0
    player_pose: np.ndarray = np.zeros(4, dtype=np.float32)
    boss_pose: np.ndarray = np.zeros(4, dtype=np.float32)
    camera_pose: np.ndarray = np.zeros(6, dtype=np.float32)
    player_animation: str = "NoAnimation"
    player_animation_duration: float = 0.
    boss_animation: str = "NoAnimation"
    boss_animation_duration: float = 0.
    combo_length: int = 0
    lock_on: bool = False

    def copy(self) -> GameState:
        """Create a copy of the ``GameState``.

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

    def as_json(self):
        """JSON encode the ``GameState`` class.

        Returns:
            The current ``GameState`` as dictionary for JSON serialization.
        """
        json_dict = self.__dict__.copy()
        for key, value in json_dict.items():
            if isinstance(value, np.ndarray):
                json_dict[key] = list(value)
        return json_dict

    @staticmethod
    def from_dict(data_dict):
        for key, value in data_dict.items():
            if isinstance(value, list):
                data_dict[key] = np.array(value)
        return GameState(**data_dict)
