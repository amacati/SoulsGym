"""Game property interface to attributes such as position, health, game loop speed etc."""
import struct
from typing import Tuple
import logging
import time

import numpy as np
from pymem.exception import MemoryReadError

from soulsgym.core.memory_manipulator import MemoryManipulator
from soulsgym.core.memory_manipulator import BASES, VALUE_ADDRESS_OFFSETS
from soulsgym.core.game_input import GameInput
from soulsgym.core.utils import wrap_to_pi

logger = logging.getLogger(__name__)


class Game:
    """Game interface class."""

    def __init__(self):
        """Initialize the MemoryManipulator which acts as an abstraction for pymem and GameInput.

        Initialization takes some time, we therefore reuse the manipulator instead of creating a new
        one for each pymem call.

        Note:
            The game has to run at this point, otherwise the initialization of MemoryManipulator
            will fail.
        """
        self.mem = MemoryManipulator()
        self._game_input = GameInput()  # Necessary for camera control etc
        self.iudex_max_hp = 1037

    def clear_cache(self):
        """Clear the resolving cache of the memory manipulator.

        Note:
            This needs to be called on every game reset and after dying.
        """
        self.mem.clear_cache()

    @property
    def player_hp(self) -> int:
        """Read the player's current hp.

        Returns:
            The player's current hit points.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerHP"], base=base)
        return self.mem.read_int(address)

    @player_hp.setter
    def player_hp(self, hp: int):
        """Set the current hit points of the player.

        Args:
            hp: The amount of hit points to set. Zeroing this value will kill the player.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerHP"], base=base)
        self.mem.write_int(address, hp)

    @property
    def player_sp(self) -> int:
        """Read the player's current sp.

        Returns:
            The player's current stamina points.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerSP"], base=base)
        return self.mem.read_int(address)

    @player_sp.setter
    def player_sp(self, sp: int):
        """Set the current stamina points of the player.

        Args:
            sp: The amount of stamina points to set.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerSP"], base=base)
        self.mem.write_int(address, sp)

    @property
    def player_max_hp(self) -> int:
        """Read the maximum player hp.

        Returns:
            The maximum hit points the player can currently have.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerMaxHP"], base=base)
        return self.mem.read_int(address)

    @player_max_hp.setter
    def player_max_hp(self, _: int):
        logger.warning("Player maximum HP can't be set. Ignoring for now")

    @property
    def player_max_sp(self) -> int:
        """Read the maximum player sp.

        Returns:
            The maximum stamina points the player can currently have.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerMaxSP"], base=base)
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
        """Read the player's current pose (position and orientation).

        The rotation is given as angle in radians as the player can only rotate around the z axis.

        Returns:
            The current player pose as [x, y, z, a].
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerA"], base=base)
        buff = self.mem.read_bytes(address, length=24)
        a, x, z, y = struct.unpack('f' + 8 * 'x' + 'fff', buff)  # Order as in the memory structure.
        return np.array([x, y, z, a])

    @player_pose.setter
    def player_pose(self, coordinates: Tuple[float]):
        """Teleport the player to given coordinates (x, y, z, a).

        Args:
            coordinates: The tuple of coordinates (x, y, z, a).
        """
        base = self.mem.base_address + BASES["B"]
        x_address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerX"], base=base)
        y_address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerY"], base=base)
        z_address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerZ"], base=base)
        a_address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerA"], base=base)
        self.gravity = False
        self.mem.write_float(x_address, coordinates[0])
        self.mem.write_float(y_address, coordinates[1])
        self.mem.write_float(z_address, coordinates[2])
        self.mem.write_float(a_address, coordinates[3])
        self.player_hp = self.player_max_hp
        self.gravity = True

    @property
    def player_animation(self) -> str:
        """Read the player's current animation.

        Returns:
            The current animation of the player as an identifier string.
        """
        # animation string has maximum of 20 chars (utf-16)
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerAnimation"], base=base)
        return self.mem.read_string(address, 40, codec="utf-16")

    @player_animation.setter
    def player_animation(self, animation: str):
        """Set the player's current animation.

        Args:
            animation: The animation identifier string.
        """
        raise NotImplementedError("Setting the player animation is not supported at the moment")

    def player_is_disabled(self):
        is_disabled = self.mem.read_bytes(BASES["PlayerDisabled"], 1)
        return is_disabled

    @property
    def player_stats(self) -> Tuple[int]:
        """Read the current player stats from the game.

        Returns:
            A Tuple with all player attributes in the same order as in the game.
        """
        base = self.mem.base_address + BASES["A"]
        sl = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["SoulLevel"], base=base))
        vigor = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vigor"], base=base))
        att = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Attunement"], base=base))
        endurance = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Endurance"], base=base))
        vit = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vitality"], base=base))
        strength = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Strength"], base=base))
        dex = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Dexterity"], base=base))
        intelligence = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Intelligence"], base=base))
        faith = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Faith"], base=base))
        luck = self.mem.read_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Luck"], base=base))
        return (sl, vigor, att, endurance, vit, strength, dex, intelligence, faith, luck)

    @player_stats.setter
    def player_stats(self, stats: Tuple[int]):
        assert len(stats) == 10, "Stats tuple dimension does not match requirements"
        base = self.mem.base_address + BASES["A"]
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["SoulLevel"], base=base),
                           stats[0])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vigor"], base=base),
                           stats[1])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Attunement"], base=base),
                           stats[2])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Endurance"], base=base),
                           stats[3])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vitality"], base=base),
                           stats[4])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Strength"], base=base),
                           stats[5])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Dexterity"], base=base),
                           stats[6])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Intelligence"], base=base), stats[7])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Faith"], base=base),
                           stats[8])
        self.mem.write_int(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Luck"], base=base),
                           stats[9])

    def check_boss_flags(self, boss_id: str) -> bool:
        """Check if the boss flags are correct.

        Args:
            boss_id: Name of the boss whose flags are checked for.

        Returns:
            True if all boss flags are correct else False.
        """
        if boss_id == "iudex":
            return self.iudex_flags
        logger.error(f"Boss name {boss_id} currently not supported")
        raise KeyError(f"Boss name {boss_id} currently not supported")

    def set_boss_flags(self, boss_id: str, flag: bool):
        """Set the boss flags of a boss.

        Args:
            boss_id: Name of the boss whose flags are set.
            flag: Value of the boss flags.

        Returns:
            True if all boss flags are correct else False.
        """
        if boss_id == "iudex":
            self.iudex_flags = flag
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_flags(self) -> bool:
        """Check whether Iudex boss flags are set correctly.

        Returns:
            True if all flags are correct, False otherwise.
        """
        base = self.mem.base_address + BASES["GameFlagData"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"], base=base)
        buff = self.mem.read_int(address)
        # The leftmost 3 bits tell if iudex is defeated(7), encountered(6) and his sword is pulled
        # out (5). We need him encountered and his sword pulled out but not defeated, so shifting
        # the bits by five has to be equal to xxxxx011 (binary). Therefore we check if the value is
        # 3 (python fills with 0 bits)
        return (buff >> 5) == 3

    @iudex_flags.setter
    def iudex_flags(self, val: bool):
        """Set Iudex flags encoutered and sword pulled out flags to `val` and defeated to 0.

        Args:
            val: Whether to set the flags or not.
        """
        flag = 1 if val else 0
        base = self.mem.base_address + BASES["GameFlagData"]
        # Encountered flag
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"], base=base)
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
            KeyError: On unsupported `boss_id`.
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
            KeyError: On unsupported `boss_id`
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
            KeyError: On unsupported `boss_id`.
        """
        if boss_id == "iudex":
            self.iudex_hp = hp
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_hp(self) -> int:
        """Read Iudex's current hp.

        Returns:
            The current hit points of Iudex.
        """
        base = self.mem.base_address + BASES["IudexA"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexHP"], base=base)
        return self.mem.read_int(address)

    @iudex_hp.setter
    def iudex_hp(self, hp: int):
        """Set Iudex's current hit points.

        Args:
            hp: The amount of hit points to set. Zeroing this value will kill Iudex.
        """
        assert 0 <= hp <= 1037  # Iudex HP has to lie in this range
        base = self.mem.base_address + BASES["IudexA"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexHP"], base=base)
        self.mem.write_int(address, hp)

    def reset_boss_hp(self, boss_id: str):
        """Reset the health points of a boss.

        Args:
            boss_id: The boss ID.

        Raises:
            KeyError: On unsupported `boss_id`.
        """
        if boss_id == "iudex":
            self.iudex_hp = self.iudex_max_hp
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss {boss_id} not supported!")

    def get_boss_pose(self, boss_id: str) -> np.ndarray:
        """Get the pose of a boss.

        Args:
            boss_id: The boss ID.

        Returns:
            The pose of the specified boss.

        Raises:
            KeyError: On unsupported `boss_id`.
        """
        if boss_id == "iudex":
            return self.iudex_pose
        logger.error(f"Boss name {boss_id} currently not supported")
        raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_pose(self) -> np.ndarray:
        """Read Iudex pose.

        Returns:
            The pose [x, y, z, a] for Iudex.
        """
        base = self.mem.base_address + BASES["IudexA"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexPoseA"], base=base)
        buff = self.mem.read_bytes(address, length=24)
        a, x, z, y = struct.unpack('f' + 8 * 'x' + 'fff', buff)  # Order as in the memory structure
        return np.array([x, y, z, a])

    @iudex_pose.setter
    def iudex_pose(self, coordinates: Tuple[float]):
        """Teleport Iudex to given coordinates (x, y, z, a).

        Args:
            coordinates: A tuple of coordinates (x, y, z, a).
        """
        game_speed = self.global_speed
        self.pause_game()
        base = self.mem.base_address + BASES["IudexA"]
        x_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexPoseX"], base=base)
        y_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexPoseY"], base=base)
        z_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexPoseZ"], base=base)
        a_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexPoseA"], base=base)
        self.mem.write_float(x_addr, coordinates[0])
        self.mem.write_float(y_addr, coordinates[1])
        self.mem.write_float(z_addr, coordinates[2])
        self.mem.write_float(a_addr, coordinates[3])
        self.global_speed = game_speed

    def get_boss_animation(self, boss_id: str) -> str:
        """Get the current animation of a boss.

        Args:
            boss_id: The boss ID.

        Returns:
            The animation of the specified boss.

        Raises:
            KeyError: On unsupported `boss_id`.
        """
        if boss_id == "iudex":
            return self.iudex_animation
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_animation(self) -> str:
        """Read Iudex's current animation.

        Returns:
            Iudex's current animation as an identifier string.
        """
        # animation string has maximum of 20 chars (utf-16)
        base = self.mem.base_address + BASES["IudexA"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexAnimation"], base=base)
        return self.mem.read_string(address, 40, codec="utf-16")

    def set_boss_attacks(self, boss_id: str, flag: bool):
        """Set the `allow_attack` flag of a boss.

        Args:
            boss_id: The boss ID.

        Raises:
            KeyError: On unsupported `boss_id`.
        """
        if boss_id == "iudex":
            self.iudex_attacks = flag
        else:
            logger.error(f"Boss name {boss_id} currently not supported")
            raise KeyError(f"Boss name {boss_id} currently not supported")

    @property
    def iudex_attacks(self) -> bool:
        """Read Iudex's attack flag.

        Returns:
            True if Iudex is allowed to attack, else False.
        """
        base = self.mem.base_address + BASES["IudexA"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexAttacks"], base=base)
        no_atk = self.mem.read_bytes(address, 1)  # Checks if attacks are forbidden
        return (no_atk[0] & 64) == 0  # Flag is saved in bit 6 (including 0)

    @iudex_attacks.setter
    def iudex_attacks(self, val: bool):
        flag = not val
        base = self.mem.base_address + BASES["IudexA"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexAttacks"], base=base)
        # Flag is saved in bit 6 (including 0)
        self.mem.write_bit(address, 6, flag)

    @property
    def camera_pose(self) -> np.ndarray:
        """Read the camera's current position and rotation.

        Returns:
            The current camera rotation as normal vector and position as coordinates
            [x, y, z, nx, ny, nz].
        """
        base = self.mem.base_address + BASES["Cam"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["CamQ1"], base=base)
        buff = self.mem.read_bytes(address, length=36)
        # cam orientation seems to be given as a normal vector for the camera plane. As with the
        # position, the game switches y and z
        _, nx, nz, ny, x, z, y = struct.unpack('f' + 4 * 'x' + 'fff' + 4 * 'x' + 'fff', buff)
        return np.array([x, y, z, nx, ny, nz])

    @camera_pose.setter
    def camera_pose(self, normal: Tuple[float]):
        """Set the camera's current orientation.

        Args:
            normal: Target vector normal to the camera plane. The reasoning behind this
                representation of orientation is that the camera never rotates around the normal
                vector.
        """
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

    def reload(self):
        """Kill the player, clear the address cache and wait for the player to respawn."""
        self.player_hp = 0
        self.resume_game()  # For safety, player might never change animation otherwise
        self.clear_cache()
        time.sleep(0.5)  # Give the game time to register player death and change animation
        while True:
            try:
                if self.player_animation == "Event63000":  # Player resurrection animation
                    break
            except (MemoryReadError, UnicodeDecodeError):  # Read during death reset might fail
                pass
            self.clear_cache()
            time.sleep(0.05)
        while self.player_animation != "Idle":  # Wait for the player to reach a safe "Idle" state
            time.sleep(0.05)

    @property
    def weapon_durability(self) -> int:
        """Read the current weapon durability.

        Returns:
            The weapon durability value.
        """
        base = self.mem.base_address + BASES["WeaponDurability"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["WeaponDurability"], base=base)
        return self.mem.read_int(address)

    @weapon_durability.setter
    def weapon_durability(self, val: int):
        assert 0 <= val <= 70, "Weapon durability has to be in [0, 70]"
        base = self.mem.base_address + BASES["WeaponDurability"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["WeaponDurability"], base=base)
        self.mem.write_int(address, val)

    @property
    def lock_on(self) -> bool:
        """Read the player's current lock on status.

        Returns:
            True if the player is currently locked on a target, else False.
        """
        base = self.mem.base_address + BASES["LockOn"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["LockOn"], base=base)
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
    def lock_on_range(self) -> float:
        """Read the current lock on range.

        Returns:
            The lock on range.
        """
        base = self.mem.base_address + BASES["LockOnParam"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["LockOnBonusRange"], base=base)
        dist = self.mem.read_float(address)
        return dist + 15  # Default lock on range is 15

    @lock_on_range.setter
    def lock_on_range(self, val: float):
        """Set the current lock on range.

        Args:
            val: Lock on range (minimum range is 15).
        """
        assert val >= 15, "Bonus lock on range must be greater or equal default (15)"
        base = self.mem.base_address + BASES["LockOnParam"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["LockOnBonusRange"], base=base)
        self.mem.write_float(address, val - 15)

    @property
    def los_lock_on_deactivate_time(self) -> float:
        """Read the current line of sight lock on deactivate time.

        Returns:
            The los lock on deactivation time.
        """
        base = self.mem.base_address + BASES["LockOnParam"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["LoSLockOnTime"], base=base)
        return self.mem.read_float(address)

    @los_lock_on_deactivate_time.setter
    def los_lock_on_deactivate_time(self, val: float):
        base = self.mem.base_address + BASES["LockOnParam"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["LoSLockOnTime"], base=base)
        self.mem.write_float(address, val)

    @property
    def global_speed(self) -> float:
        """Read the game loop speed.

        Returns:
            The game speed.
        """
        return self.mem.read_float(self.mem.base_address + BASES["GlobalSpeed"])

    @global_speed.setter
    def global_speed(self, value: float):
        """Set the game loop speed to a specific value.

        Note:
            Setting this value to 0 will effectively pause the game. Default speed is 1.

        Args:
            value: The desired speed factor.
        """
        self.mem.write_float(self.mem.base_address + BASES["GlobalSpeed"], value)
        time.sleep(0.001)  # Sleep to guarantee the game engine has reacted to the changed value

    @property
    def gravity(self) -> bool:
        """Read the current gravity activation status.

        Returns:
            True if gravity is active, else False.
        """
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["noGravity"], base=base)
        buff = self.mem.read_int(address)
        return buff & 64 == 0  # Gravity disabled flag is saved at bit 6 (including 0)

    @gravity.setter
    def gravity(self, flag):
        base = self.mem.base_address + BASES["B"]
        address = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["noGravity"], base=base)
        bit = 0 if flag else 1
        self.mem.write_bit(address, 6, bit)

    def pause_game(self):
        """Pause the game by setting the global speed to 0."""
        self.global_speed = 0

    def resume_game(self):
        """Resume the game by setting the global speed to 1."""
        self.global_speed = 1
