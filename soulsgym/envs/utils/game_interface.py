"""Game property interface to attributes such as position, health, game loop speed etc."""
import struct
from typing import Tuple
import logging
from threading import Lock
import time

import numpy as np

from soulsgym.envs.utils.memory_manipulator import MemoryManipulator
from soulsgym.envs.utils.memory_manipulator import BASES, VALUE_ADDRESS_OFFSETS
from soulsgym.envs.utils.game_input import GameInput
from soulsgym.envs.utils.utils import wrap_to_pi, Singleton

logger = logging.getLogger(__name__)


class Game(Singleton):
    """Game interface class.

    We need Game to be a Singleton in order to limit access to `get_locked_on` across all threads
    and instances.
    """
    lockon_lock = Lock()  # Protect from concurrent access to `get_locked_on()`
    last_lockon_access = None  # Time of the last call to `get_locked_on()`

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

    def clear_cache(self):
        """Clear the resolving cache of the memory manipulator.

        Note:
            This needs to be called on every game reset and after dying.
        """
        self.mem.clear_cache()

    @property
    def player_position(self) -> Tuple[float]:
        """Read the player's current position.

        Returns:
            The current player position (x,y,z,a).
        """
        buff = self.mem.read_bytes(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerA"],
                                                            base=self.mem.base_address +
                                                            BASES["B"]),
                                   length=24)
        a, x, z, y = struct.unpack('f' + 8 * 'x' + 'fff', buff)  # Order as in the memory structure.
        return x, y, z, a

    @player_position.setter
    def player_position(self, coordinates: Tuple[float]):
        """Teleport the player to given coordinates (x,y,z,a).

        Args:
            coordinates: The tuple of coordinates (x,y,z,a).
        """
        hp = self.player_hp
        base = self.mem.base_address + BASES["B"]
        x_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerX"], base=base)
        y_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerY"], base=base)
        z_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerZ"], base=base)
        a_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerA"], base=base)
        grav_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["noGravity"], base=base)

        self.mem.write_bit(grav_addr, 6,
                           1)  # Switch off gravity to avoid weird fall damage calculations
        self.mem.write_float(x_addr, coordinates[0])
        self.mem.write_float(y_addr, coordinates[1])
        self.mem.write_float(z_addr, coordinates[2])
        self.mem.write_float(a_addr, coordinates[3])
        self.mem.write_bit(grav_addr, 6, 0)  # Switch gravity back on
        self.player_hp = hp  # Sometimes player looses HP on teleport

    @property
    def target_position(self) -> Tuple[float]:
        """Read the targeted unit's current position.

        Returns:
            The position (x, y, z, a) for the targeted entity. If no entity is targeted returns
            (0, 0 , 0, 0).
        """
        buff = self.mem.read_bytes(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetX"],
                                                            base=self.mem.target_ptr),
                                   length=16)
        if buff is not None:
            x, z, y, a = struct.unpack('ffff', buff)  # Order as in the memory structure.
            return x, y, z, a
        return 0., 0., 0., 0.

    @target_position.setter
    def target_position(self, coordinates: Tuple[float]):
        """Teleport the last targeted entity to given coordinates (x,y,z,a).

        Args:
            coordinates: The tuple of coordinates (x,y,z,a).
        """
        x_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetXUpdate"],
                                          base=self.mem.target_ptr)
        y_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetYUpdate"],
                                          base=self.mem.target_ptr)
        z_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetZUpdate"],
                                          base=self.mem.target_ptr)
        a_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetAUpdate"],
                                          base=self.mem.target_ptr)
        grav_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["noGravity"],
                                             base=self.mem.target_ptr)
        self.mem.write_bit(grav_addr, 6,
                           1)  # Switch off gravity to avoid weird fall damage calculations
        self.mem.write_float(x_addr, coordinates[0])
        self.mem.write_float(y_addr, coordinates[1])
        self.mem.write_float(z_addr, coordinates[2])
        self.mem.write_float(a_addr, coordinates[3])
        self.mem.write_bit(grav_addr, 6, 0)  # Switch gravity back on

    @property
    def player_hp(self) -> int:
        """Read the player's current hp.

        Returns:
            The player's current hit points.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerHP"],
                                     base=self.mem.base_address + BASES["B"]))

    @player_hp.setter
    def player_hp(self, hp: int):
        """Set the current hit points of the player.

        Args:
            hp: The amount of hit points to set. Zeroing this value will kill the player.
        """
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerHP"],
                                     base=self.mem.base_address + BASES["B"]), hp)

    @property
    def player_sp(self) -> int:
        """Read the player's current sp.

        Returns:
            The player's current stamina points.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerSP"],
                                     base=self.mem.base_address + BASES["B"]))

    @player_sp.setter
    def player_sp(self, sp: int):
        """Set the current stamina points of the player.

        Args:
            sp: The amount of stamina points to set.
        """
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerSP"],
                                     base=self.mem.base_address + BASES["B"]), sp)

    @property
    def player_max_hp(self) -> int:
        """Read the maximum player hp.

        Returns:
            The maximum hit points the player can currently have.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerMaxHP"],
                                     base=self.mem.base_address + BASES["B"]))

    @player_max_hp.setter
    def player_max_hp(self, _: int):
        logger.warning("Player maximum HP can't be set. Ignoring for now")

    @property
    def player_max_sp(self) -> int:
        """Read the maximum player sp.

        Returns:
            The maximum stamina points the player can currently have.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerMaxSP"],
                                     base=self.mem.base_address + BASES["B"]))

    @player_max_sp.setter
    def player_max_sp(self, _: int):
        logger.warning("Player maximum SP can't be set. Ignoring for now")

    @property
    def target_hp(self) -> int:
        """Read the targeted unit's current hp.

        Returns:
            The current hit points of the last targeted entity.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedHP"], base=self.mem.target_ptr))

    @target_hp.setter
    def target_hp(self, hp: int):
        """Set the current hit points of the targeted entity.

        Args:
            hp: The amount of hit points to set. Zeroing this value will kill the targeted entity.
        """
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedHP"], base=self.mem.target_ptr),
            hp)

    @property
    def target_max_hp(self) -> int:
        """Read the targeted unit's max hp.

        Returns:
            The current maximum hit points of the last targeted entity.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedMaxHP"],
                                     base=self.mem.target_ptr))

    @target_max_hp.setter
    def target_max_hp(self, _: int):
        logger.warning("Target maximum HP can't be set. Ignoring for now")

    def check_boss_flags(self, name: str) -> bool:
        if name.lower() == "iudex":
            return self.iudex_flags
        logger.warning(f"Boss name {name} currently not supported")
        raise KeyError(f"Boss name {name} currently not supported")

    def set_boss_flags(self, name: str, flag: bool):
        if name.lower() == "iudex":
            self.iudex_flags = flag
        else:
            logger.warning(f"Boss name {name} currently not supported")
            raise KeyError(f"Boss name {name} currently not supported")

    @property
    def iudex_flags(self) -> bool:
        """Check whether Iudex boss flags are set correctly.

        Returns:
            True if all flags are correct, False otherwise.
        """
        buff = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]))
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
        # Encountered flag
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]), 5, flag)
        # Sword pulled out flag
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]), 6, flag)
        # Defeated flag
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]), 7, 0)

    @property
    def player_animation(self) -> str:
        """Read the player's current animation.

        Returns:
            The current animation of the player as an identifier string.
        """
        # animation string has maximum of 20 chars (utf-16)
        return self.mem.read_string(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerAnimation"],
                                     base=self.mem.base_address + BASES["B"]),
            40,  # noqa: E126
            codec="utf-16")

    @player_animation.setter
    def player_animation(self, animation: str):
        """Set the player's current animation.

        Args:
            animation: The animation identifier string.
        """
        self.mem.write_bytes(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerAnimation"],
                                     base=self.mem.base_address + BASES["B"]),
            bytes(animation, encoding="utf-16"))

    @property
    def player_stats(self) -> Tuple[int]:
        sl = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["SoulLevel"],
                                     base=self.mem.base_address + BASES["A"]))
        vigor = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vigor"],
                                     base=self.mem.base_address + BASES["A"]))
        att = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Attunement"],
                                     base=self.mem.base_address + BASES["A"]))
        endurance = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Endurance"],
                                     base=self.mem.base_address + BASES["A"]))
        vit = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vitality"],
                                     base=self.mem.base_address + BASES["A"]))
        strength = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Strength"],
                                     base=self.mem.base_address + BASES["A"]))
        dex = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Dexterity"],
                                     base=self.mem.base_address + BASES["A"]))
        intelligence = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Intelligence"],
                                     base=self.mem.base_address + BASES["A"]))
        faith = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Faith"],
                                     base=self.mem.base_address + BASES["A"]))
        luck = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Luck"],
                                     base=self.mem.base_address + BASES["A"]))
        return (sl, vigor, att, endurance, vit, strength, dex, intelligence, faith, luck)

    @player_stats.setter
    def player_stats(self, stats: Tuple[int]):
        assert len(stats) == 10, "Stats tuple dimension does not match requirements"
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["SoulLevel"],
                                     base=self.mem.base_address + BASES["A"]), stats[0])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vigor"],
                                     base=self.mem.base_address + BASES["A"]), stats[1])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Attunement"],
                                     base=self.mem.base_address + BASES["A"]), stats[2])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Endurance"],
                                     base=self.mem.base_address + BASES["A"]), stats[3])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Vitality"],
                                     base=self.mem.base_address + BASES["A"]), stats[4])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Strength"],
                                     base=self.mem.base_address + BASES["A"]), stats[5])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Dexterity"],
                                     base=self.mem.base_address + BASES["A"]), stats[6])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Intelligence"],
                                     base=self.mem.base_address + BASES["A"]), stats[7])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Faith"],
                                     base=self.mem.base_address + BASES["A"]), stats[8])
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["Luck"],
                                     base=self.mem.base_address + BASES["A"]), stats[9])

    @property
    def camera_pose(self) -> Tuple[float]:
        """Read the camera's current position and rotation.

        Returns:
            The current camera rotation as quarternion and position as coordinates
            (q1, q2, q3, q4, x, y, z).
        """
        buff = self.mem.read_bytes(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["CamQ1"],
                                                            base=self.mem.base_address +
                                                            BASES["Cam"]),
                                   length=36)
        # cam orientation seems to be given as a normal vector for the camera plane. As with the
        # position, the game switches y and z
        _, nx, nz, ny, x, z, y = struct.unpack('f' + 4 * 'x' + 'fff' + 4 * 'x' + 'fff', buff)
        return nx, ny, nz, x, y, z

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
        dz = self.camera_pose[2] - normal[2]
        t = 0
        # If lock on is already established and target is out of tolerances, the cam can't move. We
        # limit camera rotations to 50 actions to not run into an infinite loop where the camera
        # tries to move but can't because lock on prevents it from actually moving
        while abs(dz) > 0.05 and t < 50:
            self._game_input.single_action("cameradown" if dz > 0 else "cameraup", .02)
            dz = self.camera_pose[2] - normal[2]
            t += 1
        cpose = self.camera_pose
        d_angle = wrap_to_pi(np.arctan2(cpose[0], cpose[1]) - normal_angle)
        t = 0
        while abs(d_angle) > 0.05 and t < 50:
            self._game_input.single_action("cameraleft" if d_angle > 0 else "cameraright", .02)
            cpose = self.camera_pose
            d_angle = wrap_to_pi(np.arctan2(cpose[0], cpose[1]) - normal_angle)
            t += 1

    @property
    def target_animation(self) -> str:
        """Read the targeted unit's current animation.

        Returns:
            The current animation of the targeted entity as an identifier string.
        """
        # animation string has maximum of 20 chars (utf-16)
        return self.mem.read_string(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedAnimation"],
                                     base=self.mem.target_ptr),
            40,  # noqa: E126
            codec="utf-16")

    @target_animation.setter
    def target_animation(self, animation: str):
        """Set the targeted unit's current animation.

        Args:
            animation: The animation identifier string.
        """
        self.mem.write_bytes(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedAnimation"],
                                     base=self.mem.target_ptr), bytes(animation, encoding="utf-16"))

    @property
    def target_attacks(self):
        return self.mem.read_bytes(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetAttack"],
                                     base=self.mem.target_ptr), 1)

    @target_attacks.setter
    def target_attacks(self, val: bool):
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetAttack"],
                                     base=self.mem.target_ptr), 0, int(val))

    def get_locked_on(self) -> bool:
        """Read the player's current lock on status.

        Lock on detection works by checking if ``mem.target_ptr_volatile` is currently empty. If it
        has been written to, it has been set by a lock on event. We reset the pointer by
        deactivating and reactivating the targeted entity info for future calls to `get_locked_on`.
        This in turn means that two calls in succession almost surely return "False" since the game
        did not have the time to rewrite `mem.target_ptr_volatile`. Therefore we enforce at least
        0.1s to pass between two consecutive calls to `get_locked_on`.

        Returns:
            True if the player has currently a locked on target, else False.
        """
        with self.lockon_lock:  # Make sure `get_locked_on` hasn't been accessed in the last 0.1s
            td = time.time() - (self.last_lockon_access or 0)
            if td < 0.1:
                time.sleep(0.1 - td)
            buff = self.mem.read_int(self.mem.target_ptr_volatile)
            self.mem.deactivate_targeted_entity_info()
            self.mem.activate_targeted_entity_info()
            self.last_lockon_access = time.time()
        return buff != 0

    def reset_player_hp(self):
        """Reset the player's hit points to its current maximum."""
        self.player_hp = self.player_max_hp

    def reset_player_sp(self):
        """Reset the player's stamina points to its current maximum."""
        self.player_sp = self.player_max_sp

    def reset_target_hp(self):
        """Reset the targeted entity's hit points to its maximum."""
        self.target_hp = self.target_max_hp

    def reload(self):
        """Kill the player and clear the address cache.

        Note:
            Does not wait for the player respawn to complete.
        """
        self.clear_cache()
        self.player_hp = 0

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

    def pause_game(self):
        """Pause the game by setting the global speed to 0."""
        self.global_speed = 0

    def resume_game(self):
        """Resume the game by setting the global speed to 1."""
        self.global_speed = 1
