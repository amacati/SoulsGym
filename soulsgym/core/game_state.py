"""The ``GameState`` is a ``dataclass`` that contains all information about the game state.

It is also the observation type returned by soulsgym steps and resets and used as the internal state
representation of the gym.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import copy

import numpy as np


@dataclass
class GameState:
    """Collect all game state information in a single data class."""

    phase: int = 1
    player_hp: int = 0
    player_max_hp: int = 1
    player_sp: int = 0
    player_max_sp: int = 1
    boss_hp: int = 0
    boss_max_hp: int = 1
    player_pose: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    boss_pose: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    camera_pose: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    player_animation: str = "NoAnimation"
    player_animation_duration: float = 0.
    boss_animation: str = "NoAnimation"
    boss_animation_duration: float = 0.
    lock_on: bool = False

    def copy(self) -> GameState:
        """Create a copy of the ``GameState``.

        Returns:
            A copy of itself.
        """
        return GameState(**self.__dict__)

    def as_dict(self, deepcopy: bool = True) -> Dict:
        """Create a dictionary from the data members.

        Args:
            deepcopy: Creates a deep copy of the data. Arrays will be copied instead of referenced.

        Returns:
            The class members and their values as a dictionary.
        """
        if deepcopy:
            return copy.deepcopy(self.__dict__)
        return self.__dict__.copy()

    def as_json(self) -> Dict:
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
    def from_dict(data_dict: Dict) -> GameState:
        """Create a ``GameState`` object from a dictionary.

        Args:
            data_dict: Dictionary containing the GameState information.

        Returns:
            A GameState object with matching values.
        """
        for key, value in data_dict.items():
            if isinstance(value, list):
                data_dict[key] = np.array(value)
        return GameState(**data_dict)
