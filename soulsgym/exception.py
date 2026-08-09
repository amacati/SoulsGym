"""Exception classes for SoulsGym."""


class SoulsGymException(Exception):
    """Base class for SoulsGym exceptions."""


class GameStateError(SoulsGymException):
    """Raised when the game state deviates from expected values."""


class InvalidPlayerStateError(SoulsGymException):
    """Raised when the player state deviates from expected values."""


class InvalidBossStateError(SoulsGymException):
    """Raised when the boss state deviates from expected values."""


class ResetNeeded(SoulsGymException):
    """Raised when an environment needs to be reset, but is called with `step()`."""


class LockOnFailure(SoulsGymException):
    """Raised when lock on can't be established."""


class InjectionFailure(SoulsGymException):
    """Speedhack DLL injection failure."""


class InvalidGameSettings(SoulsGymException):
    """Raised when the game settings are invalid."""


class ResetError(SoulsGymException):
    """Raised when an environment reset fails."""


class MemoryReadError(SoulsGymException):
    """Raised when a read from the game process memory fails."""

    def __init__(self, address: int, length: int, error_code: int | None = None):
        """Build the read failure message from the address, length and OS error code."""
        super().__init__(f"Could not read {length} bytes at {address:#x} (error code {error_code})")


class MemoryWriteError(SoulsGymException):
    """Raised when a write to the game process memory fails."""

    def __init__(self, address: int, value: bytes, error_code: int | None = None):
        """Build the write failure message from the address, byte count and OS error code."""
        super().__init__(
            f"Could not write {len(value)} bytes at {address:#x} (error code {error_code})"
        )


class MemoryPatternScanError(SoulsGymException):
    """Raised when a pattern scan in the game process memory fails."""
