from soulsgym.games import Game
from soulsgym.core.game_input import GameInput
from soulsgym.core.memory_manipulator import MemoryManipulator
from soulsgym.core.speedhack import SpeedHackConnector


class EldenRing(Game):

    game_id = "EldenRing"

    def __init__(self):
        super().__init__()
        self.mem = MemoryManipulator("eldenring.exe")
        self.mem.clear_cache()  # If the singleton already exists, clear the cache
        self._game_input = GameInput("EldenRing")  # Necessary for camera control etc
        self._game_flags = {}  # Cache game flags to restore them after a game reload
        self._speed_hack_connector = SpeedHackConnector("eldenring.exe")
        self._game_speed = 1.0
        self.game_speed = 1.0

    def get_state(self):
        ...

    @property
    def player_hp(self) -> int:
        """The player's current hit points.

        Returns:
            The player's current hit points.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerHP"], base=base)
        return self.mem.read_int(address)

    @player_hp.setter
    def player_hp(self, hp: int):
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerHP"], base=base)
        self.mem.write_int(address, hp)

    @property
    def game_speed(self) -> float:
        """The game loop speed.

        Note:
            Setting this value to 0 will effectively pause the game. Default speed is 1.

        Warning:
            The process slows down with game speeds lower than 1. Values close to 0 may cause
            windows to assume the process has frozen.

        Warning:
            Values significantly higher than 1 (e.g. 5+) may not be achievable for the game loop.
            This is probably dependant on the available hardware.

        Returns:
            The game loop speed.

        Raises:
            ValueError: The game speed was set to negative values.
        """
        return self._game_speed

    @game_speed.setter
    def game_speed(self, value: float):
        if value < 0:
            raise ValueError("Attempting to set a negative game speed")
        self._speed_hack_connector.set_game_speed(value)
        self._game_speed = value
