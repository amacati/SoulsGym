"""This module contains the game interface for Elden Ring."""

import logging
import struct
import time
from functools import cached_property
from typing import Any

import numpy as np

from soulsgym.core.utils import wrap_to_pi
from soulsgym.exception import MemoryReadError
from soulsgym.games import Game

logger = logging.getLogger(__name__)


class EldenRing(Game):
    """Elden Ring game interface."""

    game_id = "EldenRing"
    process_name = "eldenring.exe"
    app_id = 1245620
    PLAYER_IDLE_ANIMATION = 10000000  # Elden Ring reports animation IDs instead of names

    def __init__(self, game_resolution: tuple[int, int] | None = None):
        """Initialize the EldenRing game interface and set the game speed to normal.

        Args:
            game_resolution: Optional game-window resolution override.
        """
        super().__init__(game_resolution=game_resolution)
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
        return self._game_window.img

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
            The game image resolution.
        """
        return self._game_window.img_resolution

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
        # The player rotation is a quaternion around the vertical axis. The game interpolates the
        # orientation towards the interpolated orientation, so we have to overwrite both
        a = coordinates[3]
        rotation = struct.pack("ffff", 0.0, np.sin(a / 2), 0.0, np.cos(a / 2))
        self.mem.write_record(self.data.addresses["PlayerLocalQ"], rotation * 2)
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
    def margit_hp(self) -> int:
        """Margit's HP."""
        return self.mem.read_record(self.data.addresses["MargitHP"])

    @margit_hp.setter
    def margit_hp(self, hp: int):
        assert 0 <= hp, "Boss HP has to be zero or positive"
        self.mem.write_record(self.data.addresses["MargitHP"], hp)

    @property
    def margit_max_hp(self) -> int:
        """Margit's maximum HP."""
        return self.mem.read_record(self.data.addresses["MargitMaxHP"])

    @margit_max_hp.setter
    def margit_max_hp(self, _: int):
        raise RuntimeError("Boss maximum HP can't be changed directly")

    def reset_boss_hp(self, boss_id: str):
        """Reset the current boss hit points.

        Args:
            boss_id: The boss ID.
        """
        setattr(self, boss_id + "_hp", getattr(self, boss_id + "_max_hp"))

    @property
    def margit_pose(self) -> np.ndarray:
        """Margit's current pose."""
        x, z, y, a = struct.unpack("ffff", self.mem.read_record(self.data.addresses["MargitXYZA"]))
        return np.array([x, y, z, a])

    @margit_pose.setter
    def margit_pose(self, coordinates: list[float]):
        assert -np.pi <= coordinates[3] <= np.pi, "Boss angle must be in [-pi, pi]"
        delta = np.array(coordinates[:3]) - self.margit_pose[:3]
        x, z, y = struct.unpack("fff", self.mem.read_record(self.data.addresses["MargitLocalXYZ"]))
        new_global_pos = struct.pack("fff", x + delta[0], z + delta[2], y + delta[1])
        self.mem.write_record(self.data.addresses["MargitLocalXYZ"], new_global_pos)

    @property
    def margit_animation(self) -> str:
        """Margit's current animation name.

        Elden Ring reports animation IDs. The static data maps the IDs of Margit's animations to
        names. IDs that are missing in the static data are reported as ``Unknown_<ID>``.

        Returns:
            The name of Margit's current animation.
        """
        animation_id = self.mem.read_record(self.data.addresses["MargitAnimation"])
        return self._boss_animation_names["margit"].get(animation_id, f"Unknown_{animation_id}")

    @margit_animation.setter
    def margit_animation(self, _: str):
        raise NotImplementedError("Boss animation can't be set!")

    @property
    def margit_animation_time(self) -> float:
        """Margit's current animation time."""
        return self.mem.read_record(self.data.addresses["MargitAnimationTime"])

    @property
    def margit_animation_max_time(self) -> float:
        """Margit's maximum animation time for the current animation."""
        return self.mem.read_record(self.data.addresses["MargitAnimationMaxTime"])

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
        self.mem.write_bit(address, index=0, value=0 if flag else 1)

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
        """Kill the player, clear the address cache and wait for the player to respawn.

        The respawn animations of Elden Ring differ from Dark Souls III. Instead of waiting for
        specific animations, we wait until the player is back ingame with positive HP and idle. The
        player is not idle while the game loads and while the player rises at the grace.
        """
        self._save_game_flags()
        self.player_hp = 0
        if self.game_speed == 0:
            self.resume()  # For safety, the player might never respawn otherwise
        self.clear_cache()
        t_start = time.time()
        while True:
            try:
                if (
                    self.is_ingame
                    and self.player_hp > 0
                    and self.player_animation == self.PLAYER_IDLE_ANIMATION
                ):
                    break
            except MemoryReadError:  # Reads during the death reset might fail
                pass
            self.clear_cache()
            time.sleep(0.05)
            if time.time() - t_start > 30:
                raise RuntimeError("Elden Ring reload timed out while waiting for player respawn")
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
        self._speed_hack_connector.update_game_speed(value)
        self._game_speed = value

    def pause(self):
        """Pause the game by setting the global speed to 0."""
        self.game_speed = 0

    def resume(self):
        """Resume the game by setting the global speed to 1."""
        self.game_speed = 1

    def clear_cache(self):
        """Clear the address cache of the game memory.

        Warning:
            The cache is invalidated on a player death and needs to be manually cleared. See
            :meth:`.GameMemory.clear_cache` for detailed information.
        """
        self.mem.clear_cache()

    def _save_game_flags(self):
        """Save game flags to the game flags cache."""
        self._game_flags["allow_player_death"] = self.allow_player_death

    def _restore_game_flags(self):
        """Set the game flags to the values saved in the game flags cache.

        Note:
            :meth:`.Game._save_game_flags` has to be called at least once before this method.
        """
        self.allow_player_death = self._game_flags["allow_player_death"]

    @cached_property
    def _boss_animation_names(self) -> dict[str, dict[int, str]]:
        """The animation names by animation ID for each boss."""
        return {
            boss: {
                animation_id: name
                for section in ("attacks", "movement", "misc")
                for name, animation_id in animations[section].items()
            }
            for boss, animations in self.data.boss_animations.items()
        }
