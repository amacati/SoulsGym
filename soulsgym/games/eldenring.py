"""This module contains the game interface for Elden Ring."""

import logging
import struct
import time
from typing import Any

import numpy as np
from pymem.exception import MemoryReadError

from soulsgym.core.utils import wrap_to_pi
from soulsgym.games import Game

logger = logging.getLogger(__name__)


class EldenRing(Game):
    """Elden Ring game interface."""

    game_id = "EldenRing"
    process_name = "eldenring.exe"

    def __init__(self):
        """Create a new EldenRing game interface and set the game speed to 1.0."""
        super().__init__()
        self._game_flags = {}  # Cache game flags to restore them after a game reload
        self._game_speed = 1.0
        self.game_speed = 1.0

    @property
    def img(self) -> np.ndarray:
        """Get the current game image as numpy array.

        Warning:
            If the game was paused (i.e. ``game_speed = 0``) before the current ```Game`` instance
            has been created, this method won't return. The game needs to be unpaused at least once
            before invoking this method.

        Images have a shape of [90, 160, 3] with RGB channels.
        """
        return self._game_window.get_img()

    @img.setter
    def img(self, _: Any):
        raise RuntimeError("Game image can't be set!")

    @property
    def img_resolution(self) -> tuple[int, int]:
        """The game image resolution.

        Note:
            This is NOT the game resolution. The game image resolution is the resolution of the
            image returned by `:meth:.EldenRing.img`.

        Returns:
            The game resolution.
        """
        return self._game_window.resolution

    @img_resolution.setter
    def img_resolution(self, resolution: tuple[int, int]):
        self._game_window.img_resolution = resolution

    @property
    def player_hp(self) -> int:
        """The player's current hit points.

        Returns:
            The player's current hit points.
        """
        return self.mem.read_record(self.data.addresses["PlayerHP"])

    @player_hp.setter
    def player_hp(self, hp: int):
        self.mem.write_record(self.data.addresses["PlayerHP"], hp)

    @property
    def player_sp(self) -> int:
        """The player's current stamina points.

        Returns:
            The player's current stamina points.
        """
        return self.mem.read_record(self.data.addresses["PlayerSP"])

    @player_sp.setter
    def player_sp(self, sp: int):
        self.mem.write_record(self.data.addresses["PlayerSP"], sp)

    @property
    def player_mp(self) -> int:
        """The player's current mana points.

        Returns:
            The player's current mana points.
        """
        return self.mem.read_record(self.data.addresses["PlayerMP"])

    @player_mp.setter
    def player_mp(self, mp: int):
        self.mem.write_record(self.data.addresses["PlayerMP"], mp)

    @property
    def player_max_hp(self) -> int:
        """The player's maximum hit points.

        Returns:
            The player's maximum hit points.
        """
        return self.mem.read_record(self.data.addresses["PlayerMaxHP"])

    @player_max_hp.setter
    def player_max_hp(self, _: int):
        raise RuntimeError("Player maximum HP can't be changed directly")

    @property
    def player_max_sp(self) -> int:
        """The player's maximum stamina points.

        Returns:
            The player's maximum stamina points.
        """
        return self.mem.read_record(self.data.addresses["PlayerMaxSP"])

    @player_max_sp.setter
    def player_max_sp(self, _: int):
        raise RuntimeError("Player maximum SP can't be changed directly")

    @property
    def player_max_mp(self) -> int:
        """The player's maximum mana points.

        Returns:
            The player's maximum mana points.
        """
        return self.mem.read_record(self.data.addresses["PlayerMaxMP"])

    @player_max_mp.setter
    def player_max_mp(self, _: int):
        raise RuntimeError("Player maximum MP can't be set directly")

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
        x, z, y, a = struct.unpack("ffff", self.mem.read_record(self.data.addresses["PlayerXYZA"]))
        return np.array([x, y, z, a])

    @player_pose.setter
    def player_pose(self, coordinates: list[float]):
        # Player coordinates have to be set in the local frame. Therefore, we first have to
        # 1) Read global coordinates
        # 2) Calculate difference to target coordinates
        # 3) Read local coordinates
        # 4) Write local coordinates + difference to memory
        assert -np.pi <= coordinates[3] <= np.pi, "Player angle must be in [-pi, pi]"
        allow_player_death = self.allow_player_death
        self.allow_player_death = False
        # Read global coordinates, calculate the difference to the target coordinates
        delta = np.array(coordinates[:3]) - self.player_pose[:3]
        # Read local coords, add the difference and write the new local coords
        x, z, y = struct.unpack("fff", self.mem.read_record(self.data.addresses["PlayerLocalXYZ"]))
        new_global_pos = struct.pack("fff", x + delta[0], z + delta[2], y + delta[1])
        self.mem.write_record(self.data.addresses["PlayerLocalXYZ"], new_global_pos)
        # TODO: Rotation is currently not working
        # address = self.mem.resolve_record(self.data.addresses["PlayerLocalQ"])
        # See https://www.euclideanspace.com/maths/geometry/rotations/conversions/index.htm
        # qw, qx, qz, qy = np.cos(coordinates[3] / 2), 0, np.sin(coordinates[3] / 2), 0
        # Order in the memory structure is qw qx qz qy
        # self.mem.write_bytes(address, struct.pack('ffff', qw, qx, qz, qy))
        self.allow_player_death = allow_player_death

    @property
    def player_animation(self) -> int:
        """The player's current animation ID.

        Note:
            The player animation cannot be overwritten.

        Returns:
            The player's current animation ID.
        """
        return self.mem.read_record(self.data.addresses["PlayerAnimation"])

    @player_animation.setter
    def player_animation(self, _: int):
        raise NotImplementedError("Setting the player animation is not supported at the moment")

    @property
    def allow_player_death(self) -> bool:
        """Disable/enable player deaths ingame."""
        return not self.mem.read_record(self.data.addresses["AllowPlayerDeath"]) & 1

    @allow_player_death.setter
    def allow_player_death(self, flag: bool):
        address = self.mem.resolve_record(self.data.addresses["AllowPlayerDeath"])
        self.mem.write_bit(address, index=0, value=0 if flag else 1)

    @property
    def player_stats(self) -> tuple[int]:
        """The current player stats from the game.

        The stats can be overwritten by a tuple of matching dimension (9) and order. Stats are
        ordered as follows: Soul Level, Vigor, Mind, Endurance, Strength, Dexterity, Intelligence,
        Faith, Arcane.

        Returns:
            A tuple with all player attributes in the same order as in the game.
        """
        address = self.mem.resolve_record(self.data.addresses["PlayerStats"])
        offsets = (0x2C, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C)
        return tuple(self.mem.read_int(address + offset) for offset in offsets)

    @player_stats.setter
    def player_stats(self, stats: list[int]):
        assert len(stats) == 9, "Stats tuple dimension does not match requirements"
        address = self.mem.resolve_record(self.data.addresses["PlayerStats"])
        offsets = (0x2C, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C)
        for stat, offset in zip(stats, offsets):
            self.mem.write_int(address + offset, stat)

    @property
    def camera_pose(self) -> np.ndarray:
        """Read the camera's current position and rotation.

        The camera orientation is specified as the normal of the camera plane. Since the plane never
        rotates around this normal the camera pose is fully specified by this 3D vector.

        Returns:
            The current camera rotation as normal vector and position as coordinates
            [x, y, z, nx, ny, nz].
        """
        buff = self.mem.read_record(self.data.addresses["LocalCam"])
        # cam orientation seems to be given as a normal vector for the camera plane. As with the
        # position, the game switches y and z
        nx, nz, ny, x, z, y = struct.unpack("fff" + 4 * "x" + "fff", buff)
        # In Elden Ring, the xyz coordinates use chunks -> We have to add the current chunk values
        cx, cz, cy = struct.unpack("fff", self.mem.read_record(self.data.addresses["ChunkCamXYZ"]))
        return np.array([x - cx, y - cy, z - cz, nx, ny, nz])

    @camera_pose.setter
    def camera_pose(self, normal: list[float]):
        assert len(normal) == 3, "Normal vector must have 3 elements"
        assert self.game_speed > 0, "Camera cannot move while the game is paused"
        normal = np.array(normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        normal_angle = np.arctan2(*normal[:2])
        cpose = self.camera_pose
        dz = cpose[5] - normal[2]  # Camera pose is [x, y, z, nx, ny, nz], we need nz
        d_angle = wrap_to_pi(np.arctan2(cpose[3], cpose[4]) - normal_angle)
        t = 0
        # If lock on is already established and target is out of tolerances, the cam can't move. We
        # limit camera rotations to 50 actions to not run into an infinite loop where the camera
        # tries to move but can't because lock on prevents it from actually moving
        while (abs(dz) > 0.1 or abs(d_angle) > 0.1) and t < 50:
            if abs(dz) > 0.1:
                self._game_input.add_action("cameradown" if dz > 0 else "cameraup")
            if abs(d_angle) > 0.1:
                self._game_input.add_action("cameraleft" if d_angle > 0 else "cameraright")
            self._game_input.update_input()
            time.sleep(0.01)
            cpose = self.camera_pose
            dz = cpose[5] - normal[2]
            d_angle = wrap_to_pi(np.arctan2(cpose[3], cpose[4]) - normal_angle)
            t += 1
            # Sometimes the initial cam key presses get "lost" and the cam does not move while the
            # buttons remain pressed. Resetting the game input on each iteration avoids this issue
            self._game_input.reset()

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
        # Get the integer ID and look up the corresponding key to this value from the bonfires dict
        int_id = self.mem.read_record(self.data.addresses["LastGrace"])
        str_id = list(self.data.bonfires.keys())[list(self.data.bonfires.values()).index(int_id)]
        return str_id

    @last_bonfire.setter
    def last_bonfire(self, name: str):
        assert name in self.data.bonfires.keys(), f"Unknown bonfire {name} specified!"
        self.mem.write_record(self.data.addresses["LastGrace"], self.data.bonfires[name])

    @property
    def lock_on(self) -> bool:
        """The player's current lock on status.

        Note:
            Lock on cannot be set.

        Returns:
            True if the player is currently locked on a target, else False.
        """
        return bool(self.mem.read_record(self.data.addresses["LockOn"])[0])

    @property
    def gravity(self) -> bool:
        """The current gravity activation status.

        Returns:
            True if gravity is active, else False.
        """
        buff = self.mem.read_record(self.data.addresses["PlayerGravity"])
        return buff & 1 == 0  # Gravity disabled flag is saved at bit 6 (including 0)

    @gravity.setter
    def gravity(self, flag: bool):
        address = self.mem.resolve_record(self.data.addresses["PlayerGravity"])
        self.mem.write_bit(address, index=0, val=0 if flag else 1)

    @property
    def allow_weapon_durability_dmg(self) -> bool:
        """Legacy parameter to comply with souls games that have weapon durability models.

        Returns:
            Always false.
        """
        return False

    @allow_weapon_durability_dmg.setter
    def allow_weapon_durability_dmg(self, _: bool): ...

    def reload(self):
        """Kill the player, clear the address cache and wait for the player to respawn."""
        raise NotImplementedError("Respawn animations need to be checked!")
        # self.player_hp = 0
        # self._save_game_flags()
        # if self.game_speed == 0:
        #    self.resume()  # For safety, player might never change animation otherwise
        # self.clear_cache()
        # self.sleep(0.5)  # Give the game time to register player death and change animation
        # while True:
        #    try:
        # Break on player resurrection animation. If missed, also break on Idle
        #        if self.player_animation in ("Event63000", "Idle"):
        #            break
        #    except (MemoryReadError, UnicodeDecodeError):  # Read during death reset might fail
        #        pass
        #    self.clear_cache()
        #    self.sleep(0.05)
        # while self.player_animation != "Idle":  # Wait for the player to reach a safe "Idle" state
        #     self.sleep(0.05)
        # self._restore_game_flags()

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
        return self.mem.read_record(self.data.addresses["Time"])

    @time.setter
    def time(self, val: int):
        assert isinstance(val, int)
        self.mem.write_record(self.data.addresses["Time"], val)

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
        return (tend - tstart) / 1000 if tend >= tstart else tend / 1000

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
