"""This module contains the game interface for Dark Souls III."""

from __future__ import annotations

import logging
import struct
import time
from typing import Any

import numpy as np
from pymem.exception import MemoryReadError

from soulsgym.core.utils import wrap_to_pi
from soulsgym.games import Game

logger = logging.getLogger(__name__)


class DarkSoulsIII(Game):
    """Dark Souls III game interface."""

    game_id = "DarkSoulsIII"
    process_name = "DarkSoulsIII.exe"

    def __init__(self):
        """Initialize the :class:`.MemoryManipulator` and the :class:`.GameInput`.

        Note:
            The game has to run at initialization, otherwise the initialization of the
            ``MemoryManipulator`` will fail.
        """
        super().__init__()  # Initialize helpers for game access and manipulation
        # Helper attributes
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
            This is NOT the game window resolution. The game window resolution is the resolution the
            game is rendered in. ``img_resolution`` is the resolution of the image returned by
            :meth:`.DarkSoulsIII.img`. Depending on the game window resolution, the image is either
            downscaled or upscaled.

        Returns:
            The game image resolution.
        """
        return self._game_window.img_resolution

    @img_resolution.setter
    def img_resolution(self, resolution: tuple[int, int]):
        self._game_window.img_resolution = resolution

    @property
    def window_resolution(self) -> tuple[int, int]:
        """The game window resolution.

        Note:
            This is NOT the image resolution. See :meth:`.DarkSoulsIII.img_resolution` for more
            details.

        Returns:
            The game window resolution.
        """
        width = self.mem.read_record(self.data.addresses["WindowScreenWidth"])
        height = self.mem.read_record(self.data.addresses["WindowScreenHeight"])
        return (width, height)

    @window_resolution.setter
    def window_resolution(self, resolution: tuple[int, int]):
        self.mem.write_record(self.data.addresses["WindowScreenWidth"], resolution[0])
        self.mem.write_record(self.data.addresses["WindowScreenHeight"], resolution[1])

    @property
    def screen_mode(self) -> str:
        """The game screen mode.

        Returns:
            The game screen mode. Either 'window' or 'fullscreen'.
        """
        mode = self.mem.read_record(self.data.addresses["ScreenMode"])
        return "window" if mode == 0 else "fullscreen"

    @screen_mode.setter
    def screen_mode(self, mode: str):
        mode = mode.lower()
        assert mode in ["window", "fullscreen"], "Screen mode must be 'window' or 'fullscreen'"
        self.mem.write_record(self.data.addresses["ScreenMode"], 0 if mode == "window" else 1)

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
    def player_max_hp(self) -> int:
        """The player's maximum hit points.

        Returns:
            The player's maximum hit points.
        """
        return self.mem.read_record(self.data.addresses["PlayerMaxHP"])

    @player_max_hp.setter
    def player_max_hp(self, _: int):
        raise NotImplementedError("Player maximum HP can't be set.")

    @property
    def player_max_sp(self) -> int:
        """The player's maximum stamina points.

        Returns:
            The player's maximum stamina points.
        """
        return self.mem.read_record(self.data.addresses["PlayerMaxSP"])

    @player_max_sp.setter
    def player_max_sp(self, _: int):
        raise NotImplementedError("Player maximum SP can't be set")

    def reset_player_hp(self):
        """Reset the player's hit points to its current maximum."""
        self.player_hp = self.player_max_hp

    def reset_player_sp(self):
        """Reset the player's stamina points to its current maximum."""
        self.player_sp = self.player_max_sp

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
        address = self.mem.resolve_record(self.data.addresses["PlayerA"])
        buff = self.mem.read_bytes(address, length=24)
        a, x, z, y = struct.unpack("f" + 8 * "x" + "fff", buff)  # Order as in the memory structure.
        return np.array([x, y, z, a])

    @player_pose.setter
    def player_pose(self, coordinates: tuple[float]):
        # If we write the x coordinate and the game loop updates the player's position immediately
        # after, we teleport before setting the other coordinates. In order to minimize these races
        # between coordinates, we pack xzy into a byte package and write it in one call. We can't
        # include `a` because of the memory layout, but this is less important as the orientation
        # can still be updated after a tick delay.
        buff_death = self.allow_player_death
        self.allow_player_death = False
        self.gravity = False
        x_address = self.mem.resolve_record(self.data.addresses["PlayerX"])
        a_address = self.mem.resolve_record(self.data.addresses["PlayerA"])
        xzy = struct.pack("fff", coordinates[0], coordinates[2], coordinates[1])  # Swap y z order
        self.mem.write_bytes(x_address, xzy)
        self.mem.write_float(a_address, coordinates[3])
        self.gravity = True
        self.allow_player_death = buff_death
        self.player_hp = self.player_max_hp

    @property
    def player_animation(self) -> str:
        """The player's current animation name.

        Note:
            The player animation cannot be overwritten.

        Returns:
            The player's current animation name.
        """
        return self.mem.read_record(self.data.addresses["PlayerAnimation"])

    @player_animation.setter
    def player_animation(self, _: str):
        raise NotImplementedError("Setting the player animation is not supported")

    @property
    def player_animation_time(self) -> float:
        """The player's current animation duration.

        Note:
            The player animation time cannot be overwritten.

        Returns:
            The player's current animation time.
        """
        return self.mem.read_record(self.data.addresses["PlayerAnimationTime"])

    @player_animation_time.setter
    def player_animation_time(self, _: float):
        raise NotImplementedError("Setting the player animation time is not supported")

    @property
    def player_animation_max_time(self) -> float:
        """The player's current animation maximum duration.

        Note:
            The player animation max time cannot be overwritten.

        Returns:
            The player's current animation maximum duration.
        """
        return self.mem.read_record(self.data.addresses["PlayerAnimationMaxTime"])

    @player_animation_max_time.setter
    def player_animation_max_time(self, _: float):
        raise NotImplementedError("Setting the player animation max time is not supported")

    @property
    def allow_player_death(self) -> bool:
        """Disable/enable player deaths ingame."""
        return self.mem.read_bytes(self.mem.bases["WorldChrManDbg_Flags"], 1) == b"\x00"

    @allow_player_death.setter
    def allow_player_death(self, flag: bool):
        self.mem.write_bytes(self.mem.bases["WorldChrManDbg_Flags"], struct.pack("B", not flag))

    @property
    def player_stats(self) -> tuple[int]:
        """The current player stats from the game.

        The stats can be overwritten by a tuple of matching dimension (10) and order. Stats are
        ordered as follows: Soul Level, Vigor, Attunement, Endurance, Vitality, Strength, Dexterity,
        Intelligence, Faith, Luck.

        Returns:
            A tuple with all player attributes in the same order as in the game.
        """
        stats_address = self.mem.resolve_record(self.data.addresses["PlayerStats"])
        # Offsets are relative to the stats address. The memory layout does not match the order of
        # the stats in the game
        offsets = (0x2C, 0x00, 0x04, 0x08, 0x28, 0x0C, 0x10, 0x14, 0x18, 0x1C)
        return tuple(self.mem.read_int(stats_address + offset) for offset in offsets)

    @player_stats.setter
    def player_stats(self, stats: tuple[int]):
        assert len(stats) == 10, "Stats tuple dimension does not match requirements"
        stats_address = self.mem.resolve_record(self.data.addresses["PlayerStats"])
        offsets = (0x2C, 0x00, 0x04, 0x08, 0x28, 0x0C, 0x10, 0x14, 0x18, 0x1C)
        for stat, offset in zip(stats, offsets):
            self.mem.write_int(stats_address + offset, stat)

    @property
    def player_frost_resistance(self) -> float:
        """The player's frostbite resistance relative to its maximum capacity.

        Is 1 if no frostbite resistance is missing, else decreases from 1. to 0.

        Returns:
            The player's frostbite resistance.
        """
        frost_resistance = self.mem.read_record(self.data.addresses["PlayerFrostResistance"])
        frost_max_resistance = self.mem.read_record(self.data.addresses["PlayerFrostResistanceMax"])
        return frost_resistance / frost_max_resistance

    @player_frost_resistance.setter
    def player_frost_resistance(self, val: float):
        assert 0 <= val <= 1, "Frostbite resistance must be between 0 and 1"
        # First, read the maximum frostbite resistance
        frost_max_resistance = self.mem.read_record(self.data.addresses["PlayerFrostResistanceMax"])
        # Calculate the absolute frostbite resistance value based on the relative value and the
        # maximum resistance
        frost_resistance = int(val * frost_max_resistance)
        self.mem.write_record(self.data.addresses["PlayerFrostResistance"], frost_resistance)

    @property
    def player_frost_effect(self) -> float:
        """The player's remaining frostbite effect duration.

        Is 0. if the player is not frostbitten, else decreases from 1. to 0.

        Returns:
            The player's frostbite effect duration.
        """
        return self.mem.read_record(self.data.addresses["PlayerFrostEffect"])

    @player_frost_effect.setter
    def player_frost_effect(self, val: float):
        assert 0 <= val <= 1, "Frostbite effect must be between 0 and 1"
        raise NotImplementedError("Setting frostbite effect is impossible with the current address")

    @property
    def iudex_flags(self) -> bool:
        """Iudex boss fight flags.

        True means the Iudex flags are set to "encountered", "sword pulled out" and "not defeated".
        In addition, the "Untended Graves" flag is set to False (0x0). All other configurations are
        False. When the flag is set to False, the "encountered", "sword pulled out" and "defeated"
        flags are set to False. The "Untended Graves" flag remains untouched.

        Note:
            Iudex Gundyr and Champion Gundyr share the exact same area in the game. Which level is
            loaded depends on the "Untended Graves" flag. If the flag is set to False, the
            "Cemetery of Ash" version of the level loads. If it is set to True (0xA), the "Untended
            Graves" version loads. The flag is set by the game whenever the player advances past
            Firelink Shrine. Warping from bonfires loads the correct level, but setting the last
            bonfire and respawning will not. Therefore, the "Untended Graves" flag has to be set to
            False when we want to spawn in "Cemetery of Ash".

        Note:
            The "Untended Graves" flag is known in the CheatEngine community as "ceremony" flag.
            Additional infos can be found in inuNorii's warp script in the grand archives cheat
            table.

        Returns:
            True if all flags are correct, False otherwise.
        """
        if self.mem.read_record(self.data.addresses["UntendedGravesFlag"]) == b"\x0a":
            return False
        # Check if the gates to Firelink Shrine are open. If they are, they have to be closed to
        # prevent the player from leaving the arena. This check might seem redundant with the Iudex
        # defeated check. However, it is possible for players to open the gates, revive Iudex with
        # the game interface and then restart the fight with open gates
        address = self.mem.resolve_record(self.data.addresses["FirelinkShrineGates"])
        if (self.mem.read_bytes(address, 1)[0] & 8) != 0:  # Gate is open, bit 3 is set
            return False
        # The leftmost 3 bits tell if iudex is defeated(7), encountered(6) and his sword is pulled
        # out (5). We need him encountered and his sword pulled out but not defeated. Therefore we
        # check if the value is 0b01100000 = 0x60
        return self.mem.read_record(self.data.addresses["IudexFlags"]) == b"\x60"  # 01100000

    @iudex_flags.setter
    def iudex_flags(self, val: bool):
        if val:
            self.mem.write_record(self.data.addresses["UntendedGravesFlag"], b"\x00")
            self.mem.write_record(self.data.addresses["IudexFlags"], b"\x60")
            address = self.mem.resolve_record(self.data.addresses["FirelinkShrineGates"])
            self.mem.write_bit(address, 3, False)  # Close the gates to Firelink Shrine if open

    @property
    def vordt_flags(self) -> bool:
        """Vordt boss fight flags.

        See :attr:`.DarkSoulsIII.iudex_flags` for more details.
        """
        return self.mem.read_record(self.data.addresses["VordtFlags"]) == b"\x40"

    @vordt_flags.setter
    def vordt_flags(self, val: bool):
        if val:
            self.mem.write_record(self.data.addresses["VordtFlags"], b"\x40")

    # We define properties for each boss. Since most code is shared between the bosses, we create
    # a property factory for each boss attribute, e.g. boss_hp. The factory takes the boss ID and
    # returns a property object for this particular boss. This allows us to define properties for
    # new bosses in a single line and reduces code duplication. Each factory assumes that the
    # addresses for bosses are stored with the boss ID as a suffix, e.g. "IudexHP" for Iudex's HP.

    def _boss_hp(boss_id: str) -> property:
        """Create a property for the boss HP given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get and set the boss HP.
        """

        @property
        def boss_hp(self: DarkSoulsIII) -> int:
            return self.mem.read_record(self.data.addresses[boss_id + "HP"])

        @boss_hp.setter
        def boss_hp(self: DarkSoulsIII, hp: int):
            assert 0 <= hp, "Boss HP has to be zero or positive"
            self.mem.write_record(self.data.addresses[boss_id + "HP"], hp)

        return boss_hp

    iudex_hp: int = _boss_hp("Iudex")
    """Iudex Gundyr's HP."""
    vordt_hp: int = _boss_hp("Vordt")
    """Vordt of the Boreal Valley's HP."""

    def _boss_max_hp(boss_id: str) -> property:
        """Create a property for the boss maximum HP given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get the boss maximum HP.
        """

        @property
        def boss_max_hp(self: DarkSoulsIII) -> int:
            return self.mem.read_record(self.data.addresses[boss_id + "MaxHP"])

        @boss_max_hp.setter
        def boss_max_hp(self: DarkSoulsIII, _: int):
            raise RuntimeError("Boss maximum HP can't be changed!")

        return boss_max_hp

    iudex_max_hp: int = _boss_max_hp("Iudex")
    """Iudex Gundyr's maximum HP."""
    vordt_max_hp: int = _boss_max_hp("Vordt")
    """Vordt of the Boreal Valley's maximum HP."""

    def reset_boss_hp(self, boss_id: str):
        """Reset the current boss hit points.

        Args:
            boss_id: The boss ID.
        """
        setattr(self, boss_id + "_hp", getattr(self, boss_id + "_max_hp"))

    def _boss_pose(boss_id: str) -> property:
        """Create a property for the boss pose given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get and set the boss pose.
        """

        @property
        def boss_pose(self: DarkSoulsIII) -> np.ndarray:
            address = self.mem.resolve_record(self.data.addresses[boss_id + "PoseA"])
            buff = self.mem.read_bytes(address, length=24)
            a, x, z, y = struct.unpack("f" + 8 * "x" + "fff", buff)  # Order as in the game memory
            return np.array([x, y, z, a])

        @boss_pose.setter
        def boss_pose(self: DarkSoulsIII, coordinates: tuple[float]):
            game_speed = self.game_speed
            self.pause()
            x_addr = self.mem.resolve_record(self.data.addresses[boss_id + "PoseX"])
            a_addr = self.mem.resolve_record(self.data.addresses[boss_id + "PoseA"])
            # Swap y and z order because the game's coordinates are stored as xzy
            xzy = struct.pack("fff", coordinates[0], coordinates[2], coordinates[1])
            # We apply the same strategy as in the player pose property to minimize data races
            self.mem.write_bytes(x_addr, xzy)
            self.mem.write_float(a_addr, coordinates[3])
            self.game_speed = game_speed

        return boss_pose

    iudex_pose: np.ndarray = _boss_pose("Iudex")
    """Iudex Gundyr's pose."""
    vordt_pose: np.ndarray = _boss_pose("Vordt")
    """Vordt of the Boreal Valley's pose."""

    def _boss_phase(boss_id: str) -> property:
        """Create a property for the boss phase given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get and set the boss phase.
        """

        @property
        def boss_phase(self: DarkSoulsIII) -> int:
            raise NotImplementedError("Boss phase detection is not implemented")

        @boss_phase.setter
        def boss_phase(self: DarkSoulsIII, phase: int):
            raise NotImplementedError("Boss phase setter is not implemented")

        return boss_phase

    iudex_phase: int = _boss_phase("Iudex")
    """Iudex Gundyr's phase."""
    vordt_phase: int = _boss_phase("Vordt")
    """Vordt of the Boreal Valley's phase."""

    def _boss_animation(boss_id: str) -> property:
        """Create a property for the boss animation given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get and set the boss animation.
        """

        @property
        def boss_animation(self: DarkSoulsIII) -> str:
            animation = self.mem.read_record(self.data.addresses[boss_id + "Animation"])
            # Damage/bleed animations 'SABlend_xxx' overwrite the current animation for ~0.4s. This
            # overwrites the actual current animation. We recover the true animation by reading two
            # registers that contain the current attack integer. This integer is -1 if no attack is
            # currently performed. In the split second between attack decisions, register 1 is
            # empty. We then read register 2. If that one is -1 as well, we default to a neutral
            # `IdleBattle` animation as a proxy for non-attacking animations. If the attack has
            # ended, SABlend has finished, and animation is a valid attack read, we still need to
            # confirm via the attack registers to not catch the tail of an animation that is already
            # finished but still lingers in animation. Alternative bleed animations are "Partxxx".
            if "SABlend" in animation or "Attack" in animation or "Part" in animation:
                address = self.mem.resolve_record(self.data.addresses[boss_id + "AttackID"])
                attack_id = self.mem.read_int(address)
                if attack_id == -1:  # Read fallback register
                    address += 0x10
                    attack_id = self.mem.read_int(address)
                    if attack_id == -1:  # No active attack, so default to best guess
                        return "IdleBattle"
                return "Attack" + str(attack_id)
            return animation

        @boss_animation.setter
        def boss_animation(self: DarkSoulsIII, _: str):
            raise NotImplementedError("Boss animation can't be set!")

        return boss_animation

    iudex_animation: str = _boss_animation("Iudex")
    """Iudex Gundyr's animation."""
    vordt_animation: str = _boss_animation("Vordt")
    """Vordt of the Boreal Valley's animation."""

    def _boss_animation_time(boss_id: str) -> property:
        """Create a property for the boss animation time given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get and set the boss animation time.
        """

        @property
        def boss_animation_time(self: DarkSoulsIII) -> float:
            return self.mem.read_record(self.data.addresses[boss_id + "AnimationTime"])

        @boss_animation_time.setter
        def boss_animation_time(self: DarkSoulsIII, _: float):
            raise NotImplementedError("Boss animation time can't be set!")

        return boss_animation_time

    iudex_animation_time: float = _boss_animation_time("Iudex")
    """Iudex Gundyr's animation time."""
    vordt_animation_time: float = _boss_animation_time("Vordt")
    """Vordt of the Boreal Valley's animation time."""

    def _boss_animation_max_time(boss_id: str) -> property:
        """Create a property for the boss animation maximum time given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            A property object that can be used to get and set the boss animation maximum time.
        """

        @property
        def boss_animation_max_time(self: DarkSoulsIII) -> float:
            return self.mem.read_record(self.data.addresses[boss_id + "AnimationMaxTime"])

        @boss_animation_max_time.setter
        def boss_animation_max_time(self: DarkSoulsIII, _: float):
            raise NotImplementedError("Boss animation max time can't be set!")

        return boss_animation_max_time

    iudex_animation_max_time: float = _boss_animation_max_time("Iudex")
    """Iudex Gundyr's animation maximum time."""
    vordt_animation_max_time: float = _boss_animation_max_time("Vordt")
    """Vordt of the Boreal Valley's animation maximum time."""

    def _boss_attacks(boss_id: str) -> property:
        """Create a property for the `boss attacks` flag given the boss ID.

        Args:
            boss_id: The boss ID.

        Returns:
            # A property object that can be used to get and set the `boss attacks` flag.
        """

        @property
        def boss_attacks(self: DarkSoulsIII) -> bool:
            return (self.mem.read_record(self.data.addresses[boss_id + "Attacks"])[0] & 64) == 0

        @boss_attacks.setter
        def boss_attacks(self: DarkSoulsIII, flag: bool):
            address = self.mem.resolve_record(self.data.addresses[boss_id + "Attacks"])
            self.mem.write_bit(address, 6, not flag)  # Flag prevents attacks if set -> invert

        return boss_attacks

    iudex_attacks: bool = _boss_attacks("Iudex")
    """Iudex Gundyr's `boss attacks` flag."""
    vordt_attacks: bool = _boss_attacks("Vordt")
    """Vordt of the Boreal Valley's `boss attacks` flag."""

    @property
    def camera_pose(self) -> np.ndarray:
        """Read the camera's current position and rotation.

        The camera orientation is specified as the normal of the camera plane. Since the plane never
        rotates around this normal the camera pose is fully specified by this 3D vector.

        Returns:
            The current camera rotation as normal vector and position as coordinates
            [x, y, z, nx, ny, nz].
        """
        address = self.mem.resolve_record(self.data.addresses["CamQx"])
        cam_buff = self.mem.read_bytes(address, length=28)
        # Cam orientation seems to be given as a normal vector for the camera plane. As with the
        # position, the game switches y and z
        nx, nz, ny, x, z, y = struct.unpack("fff" + 4 * "x" + "fff", cam_buff)
        return np.array([x, y, z, nx, ny, nz])

    @camera_pose.setter
    def camera_pose(self, normal: tuple[float]):
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
        while (abs(dz) > 0.05 or abs(d_angle) > 0.05) and t < 50:
            if abs(dz) > 0.05:
                self._game_input.add_action("cameradown" if dz > 0 else "cameraup")
            if abs(d_angle) > 0.05:
                self._game_input.add_action("cameraleft" if d_angle > 0 else "cameraright")
            self._game_input.update_input()
            time.sleep(0.02)
            cpose = self.camera_pose
            dz = cpose[5] - normal[2]
            d_angle = wrap_to_pi(np.arctan2(cpose[3], cpose[4]) - normal_angle)
            t += 1
            # Sometimes the initial cam key presses get "lost" and the cam does not move while the
            # buttons remain pressed. Resetting the game input on each iteration avoids this issue
            self._game_input.reset()

    @property
    def last_bonfire(self) -> str:
        """The bonfire name the player has rested at last.

        The bonfire name has to be in the :data:`.bonfires` dictionary.

        Returns:
            The bonfire name.
        """
        # Get the integer ID and look up the corresponding key to this value from the bonfires dict
        int_id = self.mem.read_record(self.data.addresses["LastBonfire"])
        str_id = list(self.data.bonfires.keys())[list(self.data.bonfires.values()).index(int_id)]
        return str_id

    @last_bonfire.setter
    def last_bonfire(self, name: str):
        assert name in self.data.bonfires.keys(), f"Unknown bonfire {name} specified!"
        # See Iudex flags for details on the Untended Graves flag
        ug_flag = b"\x0a" if name in ("Untended Graves", "Champion Gundyr") else b"\x00"
        self.mem.write_record(self.data.addresses["UntendedGravesFlag"], ug_flag)
        self.mem.write_record(self.data.addresses["LastBonfire"], self.data.bonfires[name])

    @property
    def allow_attacks(self) -> bool:
        """Globally enable/disable attacks for all entities."""
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xB
        return self.mem.read_bytes(address, length=1) == b"\x00"

    @allow_attacks.setter
    def allow_attacks(self, flag: bool):
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xB
        self.mem.write_bytes(address, struct.pack("B", not flag))

    @property
    def allow_hits(self) -> bool:
        """Globally enable/disable hits for all entities.

        No hits is equivalent to all entities having unlimited iframes, i.e. they are unaffected by
        all attacks, staggers etc.
        """
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xA
        return self.mem.read_bytes(address, length=1) == b"\x00"

    @allow_hits.setter
    def allow_hits(self, flag: bool):
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xA
        self.mem.write_bytes(address, struct.pack("B", not flag))

    @property
    def allow_moves(self) -> bool:
        """Globally enable/disable movement for all entities."""
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xC
        return self.mem.read_bytes(address, length=1) == b"\x00"

    @allow_moves.setter
    def allow_moves(self, flag: bool):
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xC
        self.mem.write_bytes(address, struct.pack("B", not flag))

    @property
    def allow_deaths(self) -> bool:
        """Globally enable/disable deaths for all entities."""
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0x8
        return self.mem.read_bytes(address, length=1) == b"\x00"

    @allow_deaths.setter
    def allow_deaths(self, flag: bool):
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0x8
        self.mem.write_bytes(address, struct.pack("B", not flag))

    @property
    def allow_weapon_durability_dmg(self) -> bool:
        """Globally enable/disable weapon durability damage for all entities."""
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xE
        return self.mem.read_bytes(address, length=1) == b"\x00"

    @allow_weapon_durability_dmg.setter
    def allow_weapon_durability_dmg(self, flag: bool):
        address = self.mem.bases["WorldChrManDbg_Flags"] + 0xE
        self.mem.write_bytes(address, struct.pack("B", not flag))

    def reload(self):
        """Kill the player, clear the address cache and wait for the player to respawn."""
        self.player_hp = 0
        self._save_game_flags()
        if self.game_speed == 0:
            self.resume()  # For safety, player might never change animation otherwise
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
    def lock_on(self) -> bool:
        """The player's current lock on status.

        Note:
            Lock on cannot be set.

        Returns:
            True if the player is currently locked on a target, else False.
        """
        return struct.unpack("?", self.mem.read_record(self.data.addresses["LockOn"]))[0]

    @property
    def lock_on_bonus_range(self) -> float:
        """The current maximum bonus lock on range.

        Default lock on range is 15. We only report any additional lock on range. Similarly, setting
        this value to 0 will result in the default lock on range 15.

        Returns:
            The current maximum bonus lock on range.
        """
        return self.mem.read_record(self.data.addresses["LockOnBonusRange"])

    @lock_on_bonus_range.setter
    def lock_on_bonus_range(self, val: float):
        assert val >= 0, "Bonus lock on range must be greater or equal to 0"
        self.mem.write_record(self.data.addresses["LockOnBonusRange"], val)

    @property
    def los_lock_on_deactivate_time(self) -> float:
        """The current line of sight lock on deactivate time.

        If the player looses line of sight for longer than this time period, the game will remove
        the camera lock. Default value is 2.

        Returns:
            The current line of sight lock on deactivate time.
        """
        return self.mem.read_record(self.data.addresses["LoSLockOnTime"])

    @los_lock_on_deactivate_time.setter
    def los_lock_on_deactivate_time(self, val: float):
        self.mem.write_record(self.data.addresses["LoSLockOnTime"], val)

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
            time.sleep(td)
            tcurr = self.time
            if self.timed(tcurr, tstart) > t:
                break
            td = max(t - self.timed(tcurr, tstart), 1e-3)  # 1e-3 is the min waiting interval

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

    @property
    def gravity(self) -> bool:
        """The current gravity activation status.

        Returns:
            True if gravity is active, else False.
        """
        # Gravity disabled flag is saved at bit 6 (including 0)
        return self.mem.read_record(self.data.addresses["noGravity"]) & 64 == 0

    @gravity.setter
    def gravity(self, flag: bool):
        address = self.mem.resolve_record(self.data.addresses["noGravity"])
        self.mem.write_bit(address, index=6, value=0 if flag else 1)

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

    def pause(self):
        """Pause the game by setting the global speed to 0."""
        self.game_speed = 0

    def resume(self):
        """Resume the game by setting the global speed to 1."""
        self.game_speed = 1

    def clear_cache(self):
        """Clear the address cache of the memory manipulator.

        Warning:
            The cache is invalidated on a player death and needs to be manually cleared. See
            :meth:`.MemoryManipulator.clear_cache` for detailed information.
        """
        self.mem.clear_cache()

    def _save_game_flags(self):
        """Save game flags to the game flags cache."""
        self._game_flags["allow_attacks"] = self.allow_attacks
        self._game_flags["allow_deaths"] = self.allow_deaths
        self._game_flags["allow_hits"] = self.allow_hits
        self._game_flags["allow_moves"] = self.allow_moves
        self._game_flags["allow_player_death"] = self.allow_player_death
        self._game_flags["allow_weapon_durability_dmg"] = self.allow_weapon_durability_dmg

    def _restore_game_flags(self):
        """Set the game flags to the values saved in the game flags cache.

        Note:
            :meth:`.Game._save_game_flags` has to be called at least once before this method.
        """
        self.allow_attacks = self._game_flags["allow_attacks"]
        self.allow_deaths = self._game_flags["allow_deaths"]
        self.allow_hits = self._game_flags["allow_hits"]
        self.allow_moves = self._game_flags["allow_moves"]
        self.allow_player_death = self._game_flags["allow_player_death"]
        self.allow_weapon_durability_dmg = self._game_flags["allow_weapon_durability_dmg"]
