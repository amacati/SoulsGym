"""The ``Game`` classes provide a Python interface for the game properties of the Souls games.

They abstract the memory manipulation into properties and functions that write into the appropriate
game memory addresses.

Note:
    The interface is essentially a wrapper around the :class:`.MemoryManipulator`. As such it
    inherits the same cache restrictions. See :data:`.MemoryManipulator.cache`,
    :meth:`.Game.clear_cache` and :meth:`.MemoryManipulator.clear_cache` for more information.

Warning:
    Writing into the process memory is not guaranteed to be "stable". Race conditions with the main
    game loop *will* occur and overwrite values. Coordinates are most affected by this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from soulsgym.core.memory_manipulator import MemoryManipulator
from soulsgym.core.game_input import GameInput
from soulsgym.core.game_window import GameWindow
from soulsgym.core.speedhack import SpeedHackConnector
from soulsgym.core.static import keybindings, keymap, actions, coordinates, player_animations
from soulsgym.core.static import critical_player_animations, boss_animations, player_stats
from soulsgym.core.static import bonfires, address_bases, address_offsets, address_base_patterns


@dataclass
class StaticGameData:
    """A container for the static game data.

    Only loads the static data required for the specific game to not clutter the game interface.
    """
    keybindings: dict
    keymap: dict
    actions: dict
    coordinates: dict
    player_animations: dict
    critical_player_animations: dict
    boss_animations: dict
    player_stats: dict
    bonfires: dict
    address_bases: dict
    address_offsets: dict
    address_base_patterns: dict

    def __init__(self, game_id: str):
        """Load the static data for the specific game.

        Args:
            game_id: The game ID.
        """
        self.keybindings = keybindings[game_id]
        self.keymap = keymap[game_id]
        self.actions = actions[game_id]
        self.coordinates = coordinates[game_id]
        self.player_animations = player_animations[game_id]
        self.critical_player_animations = critical_player_animations[game_id]
        self.boss_animations = boss_animations[game_id]
        self.player_stats = player_stats[game_id]
        self.bonfires = bonfires[game_id]
        self.address_bases = address_bases[game_id]
        self.address_offsets = address_offsets[game_id]
        self.address_base_patterns = address_base_patterns[game_id]


class Game(ABC):
    """Base class for all game interfaces.

    The game interface exposes the game properties as class properties and methods. Almost all
    properties and methods write directly into the game memory. The only exception is the
    :attr:`~.Game.camera_pose`. We haven't found a method to directly manipulate the camera pose
    and instead use a ``GameInput`` instance to manually control the camera with keystrokes.
    """

    def __init__(self):
        super().__init__()
        # Load the static data for the specific game
        self.data = StaticGameData(self.game_id)
        self.mem = MemoryManipulator(process_name=self.process_name)
        self.mem.clear_cache()  # If the singleton already exists, clear the cache
        self._game_window = GameWindow(self.game_id)
        self._game_input = GameInput(self.game_id)  # Necessary for camera control etc
        self._speed_hack_connector = SpeedHackConnector(self.process_name)

    @property
    @abstractmethod
    def game_id(self) -> str:
        """The game ID.

        Returns:
            The game ID.
        """

    @property
    @abstractmethod
    def process_name(self) -> str:
        """The game process name.

        Returns:
            The game process name.
        """
