import logging
import struct
from typing import Tuple
import time

from pymem.exception import MemoryReadError
import numpy as np

from soulsgym.games import Game
from soulsgym.core.game_input import GameInput
from soulsgym.core.memory_manipulator import MemoryManipulator
from soulsgym.core.speedhack import SpeedHackConnector

logger = logging.getLogger(__name__)


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
    def player_sp(self) -> int:
        """The player's current stamina points.

        Returns:
            The player's current stamina points.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerSP"], base=base)
        return self.mem.read_int(address)

    @player_sp.setter
    def player_sp(self, sp: int):
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerSP"], base=base)
        self.mem.write_int(address, sp)

    @property
    def player_mp(self) -> int:
        """The player's current mana points.

        Returns:
            The player's current mana points.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerMP"], base=base)
        return self.mem.read_int(address)

    @player_mp.setter
    def player_mp(self, mp: int):
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerMP"], base=base)
        self.mem.write_int(address, mp)

    @property
    def player_max_hp(self) -> int:
        """The player's maximum hit points.

        Returns:
            The player's maximum hit points.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerMaxHP"], base=base)
        return self.mem.read_int(address)

    @player_max_hp.setter
    def player_max_hp(self, _: int):
        logger.warning("Player maximum HP can't be set. Ignoring for now")

    @property
    def player_max_sp(self) -> int:
        """The player's maximum stamina points.

        Returns:
            The player's maximum stamina points.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerMaxSP"], base=base)
        return self.mem.read_int(address)

    @player_max_sp.setter
    def player_max_sp(self, _: int):
        logger.warning("Player maximum SP can't be set. Ignoring for now")

    @property
    def player_max_mp(self) -> int:
        """The player's maximum mana points.

        Returns:
            The player's maximum mana points.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerMaxMP"], base=base)
        return self.mem.read_int(address)

    @player_max_mp.setter
    def player_max_mp(self, _: int):
        logger.warning("Player maximum MP can't be set. Ignoring for now")

    def reset_player_hp(self):
        """Reset the player's hit points to its current maximum."""
        self.player_hp = self.player_max_hp

    def reset_player_sp(self):
        """Reset the player's stamina points to its current maximum."""
        self.player_sp = self.player_max_sp

    def reset_player_mp(self):
        """Reset the player's mana points to its current maximum."""
        self.player_mp = self.player_max_mp

    @property
    def player_pose(self) -> np.ndarray:
        """The player's current pose.

        Poses are the combination of position and rotation. In the case of game entities (e.g. the
        player or bosses) the rotation is given as a single angle in radians around the z axis.

        Setting the player's pose is more complex than just overwriting the pose values. The player
        might be killed if the teleported distance is interpreted as a fall. We save the player
        death game flag, disable player deaths and gravity, set the coordinates and restore the
        player death flag to its previous state. Gravity is always enabled after a teleport.

        Warning:
            Pose modifications are particularly affected by race conditions!

        Returns:
            The current player pose as [x, y, z, a].
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerX"], base=base)
        buff = self.mem.read_bytes(address, length=16)
        x, z, y, a = struct.unpack('ffff', buff)  # Order as in the memory structure.
        return np.array([x, y, z, a])

    @player_pose.setter
    def player_pose(self, coordinates: Tuple[float]):
        raise NotImplementedError

    @property
    def player_animation(self) -> int:
        """The player's current animation ID.

        Note:
            The player animation cannot be overwritten.

        Returns:
            The player's current animation ID.
        """
        # animation string has maximum of 20 chars (utf-16)
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerAnimation"], base=base)
        return self.mem.read_int(address)

    @player_animation.setter
    def player_animation(self, _: int):
        raise NotImplementedError("Setting the player animation is not supported at the moment")

    @property
    def allow_player_death(self) -> bool:
        """Disable/enable player deaths ingame."""
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["AllowPlayerDeath"], base=base)
        return not self.mem.read_int(address) & 1

    @allow_player_death.setter
    def allow_player_death(self, flag: bool):
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["AllowPlayerDeath"], base=base)
        bit = 0 if flag else 1
        self.mem.write_bit(address, 0, bit)

    @property
    def player_stats(self) -> Tuple[int]:
        """The current player stats from the game.

        The stats can be overwritten by a tuple of matching dimension (9) and order.

        Returns:
            A Tuple with all player attributes in the same order as in the game.
        """
        base = self.mem.bases["GameDataMan"]
        address_sl = self.mem.resolve_address(self.data.address_offsets["SoulLevel"], base=base)
        address_vigor = self.mem.resolve_address(self.data.address_offsets["Vigor"], base=base)
        address_mind = self.mem.resolve_address(self.data.address_offsets["Mind"], base=base)
        address_endurance = self.mem.resolve_address(self.data.address_offsets["Endurance"],
                                                     base=base)
        address_strength = self.mem.resolve_address(self.data.address_offsets["Strength"],
                                                    base=base)
        address_dex = self.mem.resolve_address(self.data.address_offsets["Dexterity"], base=base)
        address_intell = self.mem.resolve_address(self.data.address_offsets["Intelligence"],
                                                  base=base)
        address_faith = self.mem.resolve_address(self.data.address_offsets["Faith"], base=base)
        address_arcane = self.mem.resolve_address(self.data.address_offsets["Arcane"], base=base)
        sl = self.mem.read_int(address_sl)
        vigor = self.mem.read_int(address_vigor)
        mind = self.mem.read_int(address_mind)
        endurance = self.mem.read_int(address_endurance)
        strength = self.mem.read_int(address_strength)
        dex = self.mem.read_int(address_dex)
        intelligence = self.mem.read_int(address_intell)
        faith = self.mem.read_int(address_faith)
        arcane = self.mem.read_int(address_arcane)
        return (sl, vigor, mind, endurance, strength, dex, intelligence, faith, arcane)

    @player_stats.setter
    def player_stats(self, stats: Tuple[int]):
        assert len(stats) == 9, "Stats tuple dimension does not match requirements"
        base = self.mem.bases["GameDataMan"]
        address_sl = self.mem.resolve_address(self.data.address_offsets["SoulLevel"], base=base)
        address_vigor = self.mem.resolve_address(self.data.address_offsets["Vigor"], base=base)
        address_mind = self.mem.resolve_address(self.data.address_offsets["Mind"], base=base)
        address_endurance = self.mem.resolve_address(self.data.address_offsets["Endurance"],
                                                     base=base)
        address_strength = self.mem.resolve_address(self.data.address_offsets["Strength"],
                                                    base=base)
        address_dex = self.mem.resolve_address(self.data.address_offsets["Dexterity"], base=base)
        address_intell = self.mem.resolve_address(self.data.address_offsets["Intelligence"],
                                                  base=base)
        address_faith = self.mem.resolve_address(self.data.address_offsets["Faith"], base=base)
        address_arcane = self.mem.resolve_address(self.data.address_offsets["Arcane"], base=base)

        self.mem.write_int(address_sl, stats[0])
        self.mem.write_int(address_vigor, stats[1])
        self.mem.write_int(address_mind, stats[2])
        self.mem.write_int(address_endurance, stats[3])
        self.mem.write_int(address_strength, stats[4])
        self.mem.write_int(address_dex, stats[5])
        self.mem.write_int(address_intell, stats[6])
        self.mem.write_int(address_faith, stats[7])
        self.mem.write_int(address_arcane, stats[8])

    @property
    def is_ingame(self) -> bool:
        """Flag that checks if the player is currently loaded into the game.

        Returns:
            True if the player is ingame, else False.
        """
        try:
            return isinstance(self.player_hp, int)
        except MemoryReadError:
            return False

    @property
    def last_bonfire(self) -> int:
        """The bonfire name the player has rested at last.

        The bonfire name has to be in the :data:`.bonfires` dictionary.

        Returns:
            The bonfire name.
        """
        base = self.mem.bases["GameMan"]
        address = self.mem.resolve_address(self.data.address_offsets["LastGrace"], base=base)
        # Get the integer ID and look up the corresponding key to this value from the bonfires dict
        int_id = self.mem.read_int(address)
        str_id = list(self.data.bonfires.keys())[list(self.data.bonfires.values()).index(int_id)]
        return str_id

    @last_bonfire.setter
    def last_bonfire(self, name: str):
        assert name in self.data.bonfires.keys(), f"Unknown bonfire {name} specified!"
        base = self.mem.bases["GameMan"]
        address = self.mem.resolve_address(self.data.address_offsets["LastGrace"], base=base)
        self.mem.write_int(address, self.data.bonfires[name])

    @property
    def lock_on(self) -> bool:
        """The player's current lock on status.

        Note:
            Lock on cannot be set.

        Returns:
            True if the player is currently locked on a target, else False.
        """
        base = self.mem.bases["LockTgtMan"]
        address = self.mem.resolve_address(self.data.address_offsets["LockOn"], base=base)
        buff = self.mem.read_bytes(address, 1)[0]
        return bool(buff)

    @property
    def gravity(self) -> bool:
        """The current gravity activation status.

        Returns:
            True if gravity is active, else False.
        """
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerGravity"], base=base)
        buff = self.mem.read_int(address)
        return buff & 1 == 0  # Gravity disabled flag is saved at bit 6 (including 0)

    @gravity.setter
    def gravity(self, flag: bool):
        base = self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(self.data.address_offsets["PlayerGravity"], base=base)
        bit = 0 if flag else 1
        self.mem.write_bit(address, 0, bit)

    def reload(self):
        """Kill the player, clear the address cache and wait for the player to respawn."""
        self.player_hp = 0
        self._save_game_flags()
        if self.game_speed == 0:
            self.resume_game()  # For safety, player might never change animation otherwise
        self.clear_cache()
        self.sleep(0.5)  # Give the game time to register player death and change animation
        while True:
            try:
                # Break on player resurrection animation. If missed, also break on Idle
                if self.player_animation in ("Event63000", "Idle"):
                    break
            except (MemoryReadError, UnicodeDecodeError):  # Read during death reset might fail
                pass
            self.clear_cache()
            self.sleep(0.05)
        while self.player_animation != "Idle":  # Wait for the player to reach a safe "Idle" state
            self.sleep(0.05)
        self._restore_game_flags()

    @property
    def time(self) -> int:
        """Ingame time.

        Measured as the current game save play time in milliseconds.

        Note:
            Also increases when global game speed is set to 0, but should not increase during lags.

        Warning:
            Possibly overflows after 1193h of play time.

        Returns:
            The current game time.
        """
        base = self.mem.bases["GameDataMan"]
        address = self.mem.resolve_address(self.data.address_offsets["Time"], base=base)
        return self.mem.read_int(address)

    @time.setter
    def time(self, val: int):
        assert isinstance(val, int)
        base = self.mem.bases["GameDataMan"]
        address = self.mem.resolve_address(self.data.address_offsets["Time"], base=base)
        self.mem.write_int(address, val)

    @staticmethod
    def timed(tend: int, tstart: int) -> float:
        """Safe game time difference function.

        If time has overflowed, uses 0 as best guess for tstart. Divides by 1000 to get the time
        difference in seconds.

        Args:
            tend: End time.
            tstart: Start time.

        Returns:
            The time difference.
        """
        td = (tend - tstart) / 1000
        if td < 0:
            td = tend / 1000
        return td

    def sleep(self, t: float):
        """Custom sleep function.

        Guarantees the specified time has passed in ingame time.

        Args:
            t: Time interval in seconds.
        """
        assert t > 0
        assert self.game_speed > 0, "Game can't be paused during sleeps"
        # We save the start time and use nonbusy python sleeps while t has not been reached
        tstart, td = self.time, t
        while True:
            time.sleep(td / self.game_speed)
            tcurr = self.time
            if self.timed(tcurr, tstart) > t:
                break
            # 1e-3 / game_speed is the min waiting interval
            td = max(t - self.timed(tcurr, tstart), 1e-3) / self.game_speed

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
