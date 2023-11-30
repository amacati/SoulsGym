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
    """SpeedHack DLL injection failure."""


class InvalidGameSettings(SoulsGymException):
    """Raised when the game settings are invalid."""
