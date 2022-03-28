"""Game property interface to attributes such as position, health, game loop speed etc."""
import struct
from typing import Tuple

from soulsgym.envs.utils.memory_manipulator import MemoryManipulator
from soulsgym.envs.utils.memory_manipulator import BASES, VALUE_ADDRESS_OFFSETS


class Game:

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

    def teleport_player(self, coordinates: Tuple[float]):
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

    def teleport_target(self, coordinates: Tuple[float]):
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

    def get_player_max_hp(self) -> int:
        """Read the maximum player hp.

        Returns:
            The maximum hit points the player can currently have.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerMaxHP"],
                                     base=self.mem.base_address + BASES["B"]))

    def get_player_max_sp(self) -> int:
        """Read the maximum player sp.

        Returns:
            The maximum stamina points the player can currently have.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerMaxSP"],
                                     base=self.mem.base_address + BASES["B"]))

    def get_player_hp_sp(self) -> Tuple[int]:
        """Read the player's current hp and sp.

        Returns:
            The player's current hit points and stamina points.
        """
        buff = self.mem.read_bytes(self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerHP"],
                                                            base=self.mem.base_address +
                                                            BASES["B"]),
                                   length=28)
        return struct.unpack('i' + 20 * 'x' + 'i', buff)

    def get_target_hp(self) -> int:
        """Read the targeted unit's current hp.

        Returns:
            The current hit points of the last targeted entity.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedHP"], base=self.mem.target_ptr))

    def get_target_max_hp(self) -> int:
        """Read the targeted unit's max hp.

        Returns:
            The current maximum hit points of the last targeted entity.
        """
        return self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedMaxHP"],
                                     base=self.mem.target_ptr))

    def get_iudex_encountered(self) -> bool:
        """Check whether Iudex is already defeated.

        Returns:
            True if defeated, False otherwise.
        """
        buff = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]))
        # The leftmost 3 bits tell if iudex is defeated(7), encountered(6) and his sword is pulled
        # out (5). We need him encountered and his sword pulled out, so shifting by five has to be
        # equal to xxxxxx11 (binary), therefore we check if the value is higher than 2.
        return bool((buff >> 5) > 2)

    def get_iudex_defeated(self) -> bool:
        """Check whether Iudex is already defeated.

        Returns:
            True if defeated, False otherwise.
        """
        buff = self.mem.read_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]))
        # Bit shift by 7 since the first bit is important
        return bool(buff >> 7)

    def get_player_position(self) -> Tuple[float]:
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

    def get_player_animation(self) -> str:
        """Read the player's current animation.

        Returns:
            The current animation of the player as an identifier string.
        """
        # animation string has maximum of 20 chars (utf-16)
        return self.mem.read_string(self.mem.resolve_address(
            VALUE_ADDRESS_OFFSETS["PlayerAnimation"], base=self.mem.base_address + BASES["B"]),
                                    40,
                                    codec="utf-16")

    def set_player_animation(self, animation: str):
        """Set the player's current animation.

        Args:
            animation: The animation identifier string.
        """
        self.mem.write_bytes(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerAnimation"],
                                     base=self.mem.base_address + BASES["B"]),
            bytes(animation, encoding="utf-16"))

    def get_camera_position_rotation(self) -> Tuple[float]:
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

    def get_target_position(self) -> Tuple[float]:
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

    def get_target_animation(self) -> str:
        """Read the targeted unit's current animation.

        Returns:
            The current animation of the targeted entity as an identifier string.
        """
        # animation string has maximum of 20 chars (utf-16)
        return self.mem.read_string(self.mem.resolve_address(
            VALUE_ADDRESS_OFFSETS["TargetedAnimation"], base=self.mem.target_ptr),
                                    40,
                                    codec="utf-16")

    def set_target_animation(self, animation: str):
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

    def set_player_hp(self, hp: int):
        """Set the current hit points of the player.

        Args:
            hp: The amount of hit points to set. Zeroing this value will kill the player.
        """
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerHP"],
                                     base=self.mem.base_address + BASES["B"]), hp)

    def reset_player_hp(self):
        """Reset the player's hit points to its current maximum."""
        max_hp = self.get_player_max_hp()
        self.set_player_hp(max_hp)

    def set_player_sp(self, sp: int):
        """Set the current stamina points of the player.

        Args:
            sp: The amount of stamina points to set.
        """
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["PlayerSP"],
                                     base=self.mem.base_address + BASES["B"]), sp)

    def reset_player_sp(self):
        """Reset the player's stamina points to its current maximum."""
        max_sp = self.get_player_max_sp()
        self.set_player_sp(max_sp)

    def set_target_hp(self, hp: int):
        """Set the current hit points of the targeted entity.

        Args:
            hp: The amount of hit points to set. Zeroing this value will kill the targeted entity.
        """
        self.mem.write_int(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["TargetedHP"], base=self.mem.target_ptr),
            hp)

    def reset_target_hp(self):
        """Reset the targeted entity's hit points to its maximum."""
        max_hp = self.get_target_max_hp()
        self.set_target_hp(max_hp)

    def set_iudex_flag(self, flag: int):
        """Set the flag whether Iudex is 'defeated' (0/1).

        Always sets Iudex to 'encountered' and 'sword pulled out'.

        Args:
            flag: Boolean whether to set or unset the flag.
        """
        # Bit 7 saves the state of Iudex defeat.
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]), 5, 1)
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]), 6, 1)
        self.mem.write_bit(
            self.mem.resolve_address(VALUE_ADDRESS_OFFSETS["IudexDefeated"],
                                     base=self.mem.base_address + BASES["GameFlagData"]), 7, flag)

    def reset_iudex_and_die(self):
        """Reset Iudex and kill the player."""
        self.clear_cache()
        self.set_iudex_flag(0)
        self.set_player_hp(0)

    def set_global_speed(self, value: float):
        """Set the game speed to a specific value.

        Note:
            Setting this value to 0 will effectively pause the game. Default speed is 1.

        Args:
            value: The desired speed factor.
        """
        self.mem.write_float(self.mem.base_address + BASES["GlobalSpeed"], value)

    def pause_game(self):
        """Pause the game by setting the global speed to 0."""
        self.set_global_speed(0)

    def resume_game(self):
        """Resume the game by setting the global speed to 1."""
        self.set_global_speed(1)
