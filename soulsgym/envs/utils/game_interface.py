"""Game property interface to attributes such as position, health, game loop speed etc."""
import struct
from typing import Tuple
import logging

from soulsgym.envs.utils.memory_manipulator import MemoryManipulator
from soulsgym.envs.utils.memory_manipulator import BASES, VALUE_ADDRESS_OFFSETS

logger = logging.getLogger(__name__)


class Game:
    """Game interface class."""

    def __init__(self):
        """Initialize the MemoryManipulator which acts as an abstraction for pymem.

        Initialization takes some time, we therefore reuse the manipulator instead of creating a new
        one for each pymem call.

        Note:
            The game has to run at this point, otherwise the initialization of MemoryManipulator
            will fail.
        """
        self.mem = MemoryManipulator()

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

    @property
    def target_position(self) -> Tuple[float]:
        """Read the targeted unit's current position.

        Returns:
            The position (x, y, z, a) for the targeted entity. If no entity is targeted returns
            (0, 0 , 0, 0).
        """
        buff = self.mem.read_bytes(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedX"],
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
        x_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedX"],
                                          base=self.mem.target_ptr)
        y_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedY"],
                                          base=self.mem.target_ptr)
        z_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedZ"],
                                          base=self.mem.target_ptr)
        a_addr = self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedA"],
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

    def get_camera_pose(self) -> Tuple[float]:
        """Read the camera's current position and rotation.

        Returns:
            The current camera rotation as quarternion and position as coordinates
            (q1, q2, q3, q4, x, y, z).
        """
        buff = self.mem.read_bytes(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["CamQ1"],
                                                            base=self.mem.base_address +
                                                            BASES["D"]),
                                   length=36)
        q1, q2, q3, q4, x, z, y = struct.unpack('f' + 4 * 'x' + 'fff' + 4 * 'x' + 'fff', buff)
        return q1, q2, q3, q4, x, y, z

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

    def get_locked_on(self) -> bool:
        """Read the player's current lock on status.

        Returns:
            True if the player has currently a locked on target, else False.

        Todo:
            * Does not work correctly yet; investigate.
        """
        buff = self.mem.read_int(self.mem.target_ptr_volatile)
        self.mem.deactivate_targeted_entity_info()
        self.mem.activate_targeted_entity_info()
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
