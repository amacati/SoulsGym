"""The ``game_interface`` provides an interface for the game properties.

It abstracts the memory manipulation into properties and functions that write into the appropriate
game memory addresses.

Note:
    The ``Game`` interface is essentially a wrapper around the :class:`.MemoryManipulator`. As such
    it inherits the same cache restrictions. See :data:`.MemoryManipulator.cache`,
    :meth:`.Game.clear_cache` and :meth:`.MemoryManipulator.clear_cache` for more information.

Warning:
    Writing into the process memory is not guaranteed to be "stable". Race conditions with the main
    game loop *will* occur and overwrite values. Coordinates are most affected by this.
"""
import struct
from typing import Tuple
import logging
import time

import numpy as np
from pymem.exception import MemoryReadError

from soulsgym.core.memory_manipulator import MemoryManipulator
from soulsgym.core.game_input import GameInput
from soulsgym.core.utils import wrap_to_pi
from soulsgym.core.static import bonfires, address_bases, address_offsets

logger = logging.getLogger(__name__)


class Game:
    """The Game interface exposes the game properties as class properties and methods.

    Almost all properties and methods write directly into the game memory. The only exception is the
    :attr:`~.Game.camera_pose`. We haven't found a method to directly manipulate the camera pose
    and instead use a ``GameInput`` instance to manually control the camera with keystrokes.
    """

    def __init__(self):
        """Initialize the :class:`.MemoryManipulator` and the :class:`.GameInput`.

        Note:
            The game has to run at initialization, otherwise the initialization of the
            ``MemoryManipulator`` will fail.
        """
        self.mem = MemoryManipulator()
        self._game_input = GameInput()  # Necessary for camera control etc
        self.iudex_max_hp = 1037
        self._game_flags = {}  # Cache game flags to restore them after a game reload

    @property
    def player_hp(self) -> int:
        """The player's current hit points.

        Returns:
            The player's current hit points.
        """
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerHP"], base=base)
        return self.mem.read_int(address)

    @player_hp.setter
    def player_hp(self, hp: int):
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerHP"], base=base)
        self.mem.write_int(address, hp)

    @property
    def player_sp(self) -> int:
        """The player's current stamina points.

        Returns:
            The player's current stamina points.
        """
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerSP"], base=base)
        return self.mem.read_int(address)

    @player_sp.setter
    def player_sp(self, sp: int):
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerSP"], base=base)
        self.mem.write_int(address, sp)

    @property
    def player_max_hp(self) -> int:
        """The player's maximum hit points.

        Returns:
            The player's maximum hit points.
        """
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerMaxHP"], base=base)
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
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerMaxSP"], base=base)
        return self.mem.read_int(address)

    @player_max_sp.setter
    def player_max_sp(self, _: int):
        logger.warning("Player maximum SP can't be set. Ignoring for now")

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
        might be killed if the teleported distance is interpreted as a fall. We save all game flags,
        disable player deaths and gravity, set the coordinates and restore all flags to their
        previous state.

        Warning:
            Pose modifications are particularly affected by race conditions!

        Returns:
            The current player pose as [x, y, z, a].
        """
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerA"], base=base)
        buff = self.mem.read_bytes(address, length=24)
        a, x, z, y = struct.unpack('f' + 8 * 'x' + 'fff', buff)  # Order as in the memory structure.
        return np.array([x, y, z, a])

    @player_pose.setter
    def player_pose(self, coordinates: Tuple[float]):
        buff_death = self.allow_player_death
        self.allow_player_death = False
        buff_gravity = self.gravity
        self.gravity = False
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        x_address = self.mem.resolve_address(address_offsets["PlayerX"], base=base)
        y_address = self.mem.resolve_address(address_offsets["PlayerY"], base=base)
        z_address = self.mem.resolve_address(address_offsets["PlayerZ"], base=base)
        a_address = self.mem.resolve_address(address_offsets["PlayerA"], base=base)
        self.mem.write_float(x_address, coordinates[0])
        self.mem.write_float(y_address, coordinates[1])
        self.mem.write_float(z_address, coordinates[2])
        self.mem.write_float(a_address, coordinates[3])
        self.gravity = buff_gravity
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
        # animation string has maximum of 20 chars (utf-16)
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerAnimation"], base=base)
        return self.mem.read_string(address, 40, codec="utf-16")

    @player_animation.setter
    def player_animation(self, _: str):
        raise NotImplementedError("Setting the player animation is not supported at the moment")

    @property
    def player_animation_time(self) -> float:
        """The player's current animation duration.

        Note:
            The player animation time cannot be overwritten.

        Returns:
            The player's current animation time.
        """
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerAnimationTime"], base=base)
        return self.mem.read_float(address)

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
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["PlayerAnimationMaxTime"], base=base)
        return self.mem.read_float(address)

    @player_animation_max_time.setter
    def player_animation_max_time(self, _: float):
        raise NotImplementedError("Setting the player animation max time is not supported")

    @property
    def allow_player_death(self) -> bool:
        """Disable/enable player deaths ingame."""
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"]
        return self.mem.read_int(address) == 0

    @allow_player_death.setter
    def allow_player_death(self, flag: bool):
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"]
        self.mem.write_int(address, int(not flag))

    @property
    def player_stats(self) -> Tuple[int]:
        """The current player stats from the game.

        The stats can be overwritten by a tuple of matching dimension (10) and order.

        Returns:
            A Tuple with all player attributes in the same order as in the game.
        """
        base = self.mem.base_address + self.mem.bases["GameDataMan"]
        address_sl = self.mem.resolve_address(address_offsets["SoulLevel"], base=base)
        address_vigor = self.mem.resolve_address(address_offsets["Vigor"], base=base)
        address_att = self.mem.resolve_address(address_offsets["Attunement"], base=base)
        address_endurance = self.mem.resolve_address(address_offsets["Endurance"], base=base)
        address_vit = self.mem.resolve_address(address_offsets["Vitality"], base=base)
        address_strength = self.mem.resolve_address(address_offsets["Strength"], base=base)
        address_dex = self.mem.resolve_address(address_offsets["Dexterity"], base=base)
        address_intell = self.mem.resolve_address(address_offsets["Intelligence"], base=base)
        address_faith = self.mem.resolve_address(address_offsets["Faith"], base=base)
        address_luck = self.mem.resolve_address(address_offsets["Luck"], base=base)
        sl = self.mem.read_int(address_sl)
        vigor = self.mem.read_int(address_vigor)
        att = self.mem.read_int(address_att)
        endurance = self.mem.read_int(address_endurance)
        vit = self.mem.read_int(address_vit)
        strength = self.mem.read_int(address_strength)
        dex = self.mem.read_int(address_dex)
        intelligence = self.mem.read_int(address_intell)
        faith = self.mem.read_int(address_faith)
        luck = self.mem.read_int(address_luck)
        return (sl, vigor, att, endurance, vit, strength, dex, intelligence, faith, luck)

    @player_stats.setter
    def player_stats(self, stats: Tuple[int]):
        assert len(stats) == 10, "Stats tuple dimension does not match requirements"
        base = self.mem.base_address + self.mem.bases["GameDataMan"]
        address_sl = self.mem.resolve_address(address_offsets["SoulLevel"], base=base)
        address_vigor = self.mem.resolve_address(address_offsets["Vigor"], base=base)
        address_att = self.mem.resolve_address(address_offsets["Attunement"], base=base)
        address_endurance = self.mem.resolve_address(address_offsets["Endurance"], base=base)
        address_vit = self.mem.resolve_address(address_offsets["Vitality"], base=base)
        address_strength = self.mem.resolve_address(address_offsets["Strength"], base=base)
        address_dex = self.mem.resolve_address(address_offsets["Dexterity"], base=base)
        address_intell = self.mem.resolve_address(address_offsets["Intelligence"], base=base)
        address_faith = self.mem.resolve_address(address_offsets["Faith"], base=base)
        address_luck = self.mem.resolve_address(address_offsets["Luck"], base=base)

        self.mem.write_int(address_sl, stats[0])
        self.mem.write_int(address_vigor, stats[1])
        self.mem.write_int(address_att, stats[2])
        self.mem.write_int(address_endurance, stats[3])
        self.mem.write_int(address_vit, stats[4])
        self.mem.write_int(address_strength, stats[5])
        self.mem.write_int(address_dex, stats[6])
        self.mem.write_int(address_intell, stats[7])
        self.mem.write_int(address_faith, stats[8])
        self.mem.write_int(address_luck, stats[9])

    def check_boss_flags(self, boss_id: str) -> bool:
        """Check if the boss flags are correct for starting the boss fight.

        Args:
            boss_id: The boss ID.

        Returns:
            True if all boss flags are correct else False.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            return self.iudex_flags
        logger.error(f"Boss name {boss_id} currently not supported")
        raise KeyError(f"Boss name {boss_id} currently not supported")

    def set_boss_flags(self, boss_id: str, flag: bool):
        """Set the boss flags of a boss to enable the boss fight.

        Args:
            boss_id: The boss ID.
            flag: Value of the boss flags.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            self.iudex_flags = flag
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_flags(self) -> bool:
        """Iudex boss fight flags.

        True means the Iudex flags are set to "encountered", "sword pulled out" and "not defeated".
        All other configurations are False. When the flag is set to False, the "encountered",
        "sword pulled out" and "defeated" flags are set to False.

        Returns:
            True if all flags are correct, False otherwise.
        """
        base = self.mem.base_address + self.mem.bases["GameFlagData"]
        address = self.mem.resolve_address(address_offsets["IudexDefeated"], base=base)
        buff = self.mem.read_int(address)
        # The leftmost 3 bits tell if iudex is defeated(7), encountered(6) and his sword is pulled
        # out (5). We need him encountered and his sword pulled out but not defeated, so shifting
        # the bits by five has to be equal to xxxxx011 (binary). Therefore we check if the value is
        # 3 (python fills with 0 bits)
        return (buff >> 5) == 3

    @iudex_flags.setter
    def iudex_flags(self, val: bool):
        flag = 1 if val else 0
        base = self.mem.base_address + self.mem.bases["GameFlagData"]
        # Encountered flag
        address = self.mem.resolve_address(address_offsets["IudexDefeated"], base=base)
        self.mem.write_bit(address, 5, flag)
        # Sword pulled out flag
        self.mem.write_bit(address, 6, flag)
        # Defeated flag
        self.mem.write_bit(address, 7, 0)

    def get_boss_max_hp(self, boss_id: str) -> int:
        """Get the maximum health points of a boss.

        Args:
            boss_id: The boss ID.

        Returns:
            The maximum health points of the specified boss.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            return self.iudex_max_hp
        logger.error(f"Boss name {boss_id} currently not supported")
        raise KeyError(f"Boss name {boss_id} currently not supported")

    def get_boss_hp(self, boss_id: str) -> int:
        """Get the health points of a boss.

        Args:
            boss_id: The boss ID.

        Returns:
            The health points of the specified boss.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            return self.iudex_hp
        logger.error(f"Boss name {boss_id} currently not supported")
        raise KeyError(f"Boss name {boss_id} currently not supported")

    def set_boss_hp(self, boss_id: str, hp: int):
        """Set the health points of a boss.

        Args:
            boss_id: The boss ID.
            hp: The health points assigned to the boss.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            self.iudex_hp = hp
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_hp(self) -> int:
        """Iudex Gundyr's current hit points.

        Returns:
            Iudex Gundyr's current hit points.
        """
        base = self.mem.base_address + address_bases["Iudex"]
        address = self.mem.resolve_address(address_offsets["IudexHP"], base=base)
        return self.mem.read_int(address)

    @iudex_hp.setter
    def iudex_hp(self, hp: int):
        assert 0 <= hp <= 1037  # Iudex HP has to lie in this range
        base = self.mem.base_address + address_bases["Iudex"]
        address = self.mem.resolve_address(address_offsets["IudexHP"], base=base)
        self.mem.write_int(address, hp)

    def reset_boss_hp(self, boss_id: str):
        """Reset the current boss hit points.

        Args:
            boss_id: The boss ID.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            self.iudex_hp = self.iudex_max_hp
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss {boss_id} not supported!")

    def get_boss_pose(self, boss_id: str) -> np.ndarray:
        """Get the current boss pose.

        Args:
            boss_id: The boss ID.

        Returns:
            The current boss pose.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            return self.iudex_pose
        logger.error(f"Boss name {boss_id} currently not supported")
        raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_pose(self) -> np.ndarray:
        """Iudex Gundyr's pose.

        Returns:
            Iudex Gundyr's pose.
        """
        base = self.mem.base_address + address_bases["Iudex"]
        address = self.mem.resolve_address(address_offsets["IudexPoseA"], base=base)
        buff = self.mem.read_bytes(address, length=24)
        a, x, z, y = struct.unpack('f' + 8 * 'x' + 'fff', buff)  # Order as in the memory structure
        return np.array([x, y, z, a])

    @iudex_pose.setter
    def iudex_pose(self, coordinates: Tuple[float]):
        game_speed = self.global_speed  # TODO: Verify this is necessary
        self.pause_game()
        base = self.mem.base_address + address_bases["Iudex"]
        x_addr = self.mem.resolve_address(address_offsets["IudexPoseX"], base=base)
        y_addr = self.mem.resolve_address(address_offsets["IudexPoseY"], base=base)
        z_addr = self.mem.resolve_address(address_offsets["IudexPoseZ"], base=base)
        a_addr = self.mem.resolve_address(address_offsets["IudexPoseA"], base=base)
        self.mem.write_float(x_addr, coordinates[0])
        self.mem.write_float(y_addr, coordinates[1])
        self.mem.write_float(z_addr, coordinates[2])
        self.mem.write_float(a_addr, coordinates[3])
        self.global_speed = game_speed

    def get_boss_phase(self, boss_id: str) -> int:
        """Get the current boss phase.

        boss_id: The boss ID.

        Returns:
            The current boss phase.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            return self.iudex_phase
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_phase(self) -> int:
        """Phase detection is currently not implemented."""
        raise NotImplementedError

    def get_boss_animation(self, boss_id: str) -> str:
        """Get the current boss animation.

        Args:
            boss_id: The boss ID.

        Returns:
            The current boss animation.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            return self.iudex_animation
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_animation(self) -> str:
        """Iudex Gundyr's current animation.

        Returns:
            Iudex Gundyr's current animation.
        """
        # Animation string has maximum of 20 chars (utf-16)
        base = self.mem.base_address + address_bases["Iudex"]
        address = self.mem.resolve_address(address_offsets["IudexAnimation"], base=base)
        animation = self.mem.read_string(address, 40, codec="utf-16")
        # Damage animations 'SABlend_xxx' overwrite the current animation for ~0.4s. This leads to
        # bad info since the boss' true animation cannot infered. We recover the true animation by
        # reading two registers that contain the current attack integer. This integer is -1 if no
        # attack is currently performed. In the split second between attack decisions, register 1 is
        # empty. We then read register 2. If that one is -1 as well, we default to a neutral
        # `IdleBattle` animation, but this could really be any non-attacking animation.
        # If the attack has ended, SABlend has finished and animation is a valid attack read we
        # still need to confirm via the attack registers to not catch the tail of an animation that
        # is already finished but still lingers in animation.
        if "SABlend" in animation or "Attack" in animation:
            base = self.mem.base_address + address_bases["Iudex"]
            address = self.mem.resolve_address(address_offsets["IudexAttackID"], base=base)
            attack_id = self.mem.read_int(address)
            if attack_id == -1:  # Read fallback register
                address += 0x10
                attack_id = self.mem.read_int(address)
                if attack_id == -1:  # No active attack, so default to best guess
                    return "IdleBattle"
            return "Attack" + str(attack_id)
        return animation

    def set_boss_attacks(self, boss_id: str, flag: bool):
        """Set the ``allow_attack`` flag of a boss.

        Args:
            boss_id: The boss ID.

        Raises:
            KeyError: ``boss_id`` does not match any known boss.
        """
        if boss_id == "iudex":
            self.iudex_attacks = flag
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_attacks(self) -> bool:
        """Iudex Gundyr's ``allow_attack`` flag.

        Returns:
            True if Iudex is allowed to attack, else False.
        """
        base = self.mem.base_address + address_bases["Iudex"]
        address = self.mem.resolve_address(address_offsets["IudexAttacks"], base=base)
        no_atk = self.mem.read_bytes(address, 1)  # Checks if attacks are forbidden
        return (no_atk[0] & 64) == 0  # Flag is saved in bit 6 (including 0)

    @iudex_attacks.setter
    def iudex_attacks(self, val: bool):
        flag = not val
        base = self.mem.base_address + address_bases["Iudex"]
        address = self.mem.resolve_address(address_offsets["IudexAttacks"], base=base)
        # Flag is saved in bit 6 (including 0)
        self.mem.write_bit(address, 6, flag)

    @property
    def camera_pose(self) -> np.ndarray:
        """Read the camera's current position and rotation.

        The camera orientation is specified as the normal of the camera plane. Since the plane never
        rotates around this normal the camera pose is fully specified by this 3D vector.

        Returns:
            The current camera rotation as normal vector and position as coordinates
            [x, y, z, nx, ny, nz].
        """
        base = self.mem.base_address + address_bases["Cam"]
        address = self.mem.resolve_address(address_offsets["CamQ1"], base=base)
        buff = self.mem.read_bytes(address, length=36)
        # cam orientation seems to be given as a normal vector for the camera plane. As with the
        # position, the game switches y and z
        _, nx, nz, ny, x, z, y = struct.unpack('f' + 4 * 'x' + 'fff' + 4 * 'x' + 'fff', buff)
        return np.array([x, y, z, nx, ny, nz])

    @camera_pose.setter
    def camera_pose(self, normal: Tuple[float]):
        assert len(normal) == 3, "Normal vector must have 3 elements"
        normal = np.array(normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        normal_angle = np.arctan2(*normal[:2])
        dz = self.camera_pose[5] - normal[2]  # Camera pose is [x, y, z, nx, ny, nz], we need nz
        t = 0
        # If lock on is already established and target is out of tolerances, the cam can't move. We
        # limit camera rotations to 50 actions to not run into an infinite loop where the camera
        # tries to move but can't because lock on prevents it from actually moving
        while abs(dz) > 0.05 and t < 50:
            self._game_input.single_action("cameradown" if dz > 0 else "cameraup", .02)
            dz = self.camera_pose[5] - normal[2]
            t += 1
        cpose = self.camera_pose
        d_angle = wrap_to_pi(np.arctan2(cpose[3], cpose[4]) - normal_angle)
        t = 0
        while abs(d_angle) > 0.05 and t < 50:
            self._game_input.single_action("cameraleft" if d_angle > 0 else "cameraright", .02)
            cpose = self.camera_pose
            d_angle = wrap_to_pi(np.arctan2(cpose[3], cpose[4]) - normal_angle)
            t += 1

    @property
    def last_bonfire(self) -> str:
        """The bonfire name the player has rested at last.

        The bonfire name has to be in the :data:`.bonfires` dictionary.

        Returns:
            The bonfire name.
        """
        base = self.mem.base_address + self.mem.bases["GameMan"]
        address = self.mem.resolve_address(address_offsets["LastBonfire"], base=base)
        buff = self.mem.read_bytes(address, 4)
        # Get the integer ID and look up the corresponding key to this value from the bonfires dict
        int_id = int.from_bytes(buff, byteorder="little")
        str_id = list(bonfires.keys())[list(bonfires.values()).index(int_id)]
        return str_id

    @last_bonfire.setter
    def last_bonfire(self, name: str):
        assert name in bonfires.keys(), f"Unknown bonfire {name} specified!"
        base = self.mem.base_address + self.mem.bases["GameMan"]
        address = self.mem.resolve_address(address_offsets["LastBonfire"], base=base)
        buff = (bonfires[name]).to_bytes(4, byteorder='little')
        self.mem.write_bytes(address, buff)

    @property
    def allow_attacks(self) -> bool:
        """Globally enable/disable attacks for all entities."""
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xB
        return self.mem.read_int(address) == 0

    @allow_attacks.setter
    def allow_attacks(self, flag: bool):
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xB
        self.mem.write_bytes(address, struct.pack('B', not flag))

    @property
    def allow_hits(self) -> bool:
        """Globally enable/disable hits for all entities.

        No hits is equivalent to all entities having unlimited iframes, i.e. they are unaffected by
        all attacks, staggers etc.
        """
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xA
        return self.mem.read_int(address) == 0

    @allow_hits.setter
    def allow_hits(self, flag: bool):
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xA
        self.mem.write_bytes(address, struct.pack('B', not flag))

    @property
    def allow_moves(self) -> bool:
        """Globally enable/disable movement for all entities."""
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xC
        return self.mem.read_int(address) == 0

    @allow_moves.setter
    def allow_moves(self, flag: bool):
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xC
        self.mem.write_bytes(address, struct.pack('B', not flag))

    @property
    def allow_deaths(self) -> bool:
        """Globally enable/disable deaths for all entities."""
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0x8
        return self.mem.read_int(address) == 0

    @allow_deaths.setter
    def allow_deaths(self, flag: bool):
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0x8
        self.mem.write_bytes(address, struct.pack('B', not flag))

    @property
    def allow_weapon_durability_dmg(self) -> bool:
        """Globally enable/disable weapon durability damage for all entities."""
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xE
        return self.mem.read_int(address) == 0

    @allow_weapon_durability_dmg.setter
    def allow_weapon_durability_dmg(self, flag: bool):
        address = self.mem.base_address + self.mem.bases["WorldChrManDbg_Flags"] + 0xE
        self.mem.write_bytes(address, struct.pack('B', not flag))

    def reload(self):
        """Kill the player, clear the address cache and wait for the player to respawn."""
        self.player_hp = 0
        self._save_game_flags()
        self.resume_game()  # For safety, player might never change animation otherwise
        self.clear_cache()
        self.game.sleep(0.5)  # Give the game time to register player death and change animation
        while True:
            try:
                if self.player_animation == "Event63000":  # Player resurrection animation
                    break
            except (MemoryReadError, UnicodeDecodeError):  # Read during death reset might fail
                pass
            self.clear_cache()
            self.game.sleep(0.05)
        while self.player_animation != "Idle":  # Wait for the player to reach a safe "Idle" state
            self.game.sleep(0.05)
        self._restore_game_flags()

    @property
    def lock_on(self) -> bool:
        """The player's current lock on status.

        Note:
            Lock on cannot be set.

        Returns:
            True if the player is currently locked on a target, else False.
        """
        base = self.mem.base_address + self.mem.bases["LockTgtMan"]
        address = self.mem.resolve_address(address_offsets["LockOn"], base=base)
        buff = self.mem.read_bytes(address, 1)
        lock_on = struct.unpack("?", buff)[0]  # Interpret buff as boolean
        # We suspect the lock on flag actually signals the alignment of the target with the camera.
        # If lock_on is False, we therefore wait a small amount of time and recheck to make sure we
        # don't get too many False negatives
        if not lock_on:
            time.sleep(0.01)
            buff = self.mem.read_bytes(address, 1)
            lock_on = struct.unpack("?", buff)[0]
        return lock_on

    @property
    def lock_on_bonus_range(self) -> float:
        """The current maximum bonus lock on range.

        Default lock on range is 15. We only report any additional lock on range. Similarly, setting
        this value to 0 will result in the default lock on range 15.

        Returns:
            The current maximum bonus lock on range.
        """
        base = self.mem.base_address + self.mem.bases["LockTgtMan"]
        address = self.mem.resolve_address(address_offsets["LockOnBonusRange"], base=base)
        dist = self.mem.read_float(address)
        return dist

    @lock_on_bonus_range.setter
    def lock_on_bonus_range(self, val: float):
        assert val >= 0, "Bonus lock on range must be greater or equal to 0"
        base = self.mem.base_address + self.mem.bases["LockTgtMan"]
        address = self.mem.resolve_address(address_offsets["LockOnBonusRange"], base=base)
        self.mem.write_float(address, val)

    @property
    def los_lock_on_deactivate_time(self) -> float:
        """The current line of sight lock on deactivate time.

        If the player looses line of sight for longer than this time period, the game will remove
        the camera lock. Default value is 2.

        Returns:
            The current line of sight lock on deactivate time.
        """
        base = self.mem.base_address + self.mem.bases["LockTgtMan"]
        address = self.mem.resolve_address(address_offsets["LoSLockOnTime"], base=base)
        return self.mem.read_float(address)

    @los_lock_on_deactivate_time.setter
    def los_lock_on_deactivate_time(self, val: float):
        base = self.mem.base_address + self.mem.bases["LockTgtMan"]
        address = self.mem.resolve_address(address_offsets["LoSLockOnTime"], base=base)
        self.mem.write_float(address, val)

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
        base = self.mem.base_address + self.mem.bases["GameDataMan"]
        address = self.mem.resolve_address(address_offsets["Time"], base=base)
        return self.mem.read_int(address)

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
        # We save the start time and use nonbusy python sleeps while t has not been reached
        tstart, td = self.time, t
        while True:
            time.sleep(td)
            tcurr = self.time
            if self.timed(tcurr, tstart) > t:
                break
            # 1e-3 is the min waiting interval
            td = max(t - self.timed(tcurr, tstart), 1e-3)

    @property
    def global_speed(self) -> float:
        """The game loop speed.

        Note:
            Setting this value to 0 will effectively pause the game. Default speed is 1.

        Returns:
            The game loop speed.
        """
        return self.mem.read_float(self.mem.base_address + self.mem.bases["GlobalSpeed"])

    @global_speed.setter
    def global_speed(self, value: float):
        self.mem.write_float(self.mem.base_address + self.mem.bases["GlobalSpeed"], value)
        time.sleep(0.001)  # Increase robustness by giving the game time to write the value

    @property
    def gravity(self) -> bool:
        """The current gravity activation status.

        Returns:
            True if gravity is active, else False.
        """
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["noGravity"], base=base)
        buff = self.mem.read_int(address)
        return buff & 64 == 0  # Gravity disabled flag is saved at bit 6 (including 0)

    @gravity.setter
    def gravity(self, flag: bool):
        base = self.mem.base_address + self.mem.bases["WorldChrMan"]
        address = self.mem.resolve_address(address_offsets["noGravity"], base=base)
        bit = 0 if flag else 1
        self.mem.write_bit(address, 6, bit)

    def pause_game(self):
        """Pause the game by setting the global speed to 0."""
        self.global_speed = 0

    def resume_game(self):
        """Resume the game by setting the global speed to 1."""
        self.global_speed = 1

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
