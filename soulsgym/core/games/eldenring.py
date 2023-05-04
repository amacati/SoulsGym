from soulsgym.core.games import Game
from soulsgym.core.game_input import GameInput
from soulsgym.core.memory_manipulator import MemoryManipulator


class EldenRing(Game):

    def __init__(self):
        self.mem = MemoryManipulator()
        self.mem.clear_cache()  # If the singleton already exists, clear the cache
        self._game_input = GameInput()  # Necessary for camera control etc
