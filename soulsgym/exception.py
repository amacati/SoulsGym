"""Exception classes for SoulsGym."""


class GameStateError(Exception):
    """Raised when the game state deviates from expected values."""


class InvalidPlayerStateError(Exception):
    """Raised when the player state deviates from expected values."""


class InvalidBossStateError(Exception):
    """Raised when the boss state deviates from expected values."""


class ResetNeeded(Exception):
    """Raised when an environment needs to be reset, but is called with `step()`."""


class LockOnFailure(Exception):
    """Raised when lock on can't be established."""
