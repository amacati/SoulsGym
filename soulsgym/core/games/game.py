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
from abc import ABC


class Game(ABC):
    """Base class for all game interfaces.

    The game interface exposes the game properties as class properties and methods. Almost all
    properties and methods write directly into the game memory. The only exception is the
    :attr:`~.Game.camera_pose`. We haven't found a method to directly manipulate the camera pose
    and instead use a ``GameInput`` instance to manually control the camera with keystrokes.
    """
