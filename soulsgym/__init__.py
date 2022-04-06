"""Gym registration and module setup."""
from pathlib import Path
import logging

from gym.envs.registration import register
import win32api

logger = logging.getLogger(__name__)


def _check_ds3_path() -> bool:
    # DarkSoulsIII is not registered in Window's registry, therefore we look for the path itself
    ds3path = Path()
    drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
    steam_path = Path("Program Files (x86)") / "Steam" / "steamapps" / "common" / "DARK SOULS III" \
        / "Game" / "DarkSoulsIII.exe"
    for drive in drives:
        ds3path = Path(drive) / steam_path
        if ds3path.is_file():
            return True
    return False


if not _check_ds3_path():
    logger.warning("Could not find Dark Souls III executable. Continuing for now...")


def set_log_level(level: int):
    """Set log level for the soulsgym module.

    Args:
        level: Logger level for the module. Uses the standard library logging module levels.
    """
    logger.setLevel(level)


# Register environments in OpenAI gym
register(id="SoulsGymIudex-v0",
         entry_point='soulsgym.envs.iudex_env:IudexEnv',
         nondeterministic=True)
