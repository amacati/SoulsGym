"""The ``soulsgym`` package is a collection of OpenAI gym environments for Dark Souls III.

It contains two main components. The first module :mod:`soulsgym.envs` includes a core environment
as well as an individual environment for each available boss fight. SoulsGym uses Dark Souls III as
the underlying engine that is modified at runtime by reading and writing into the game memory to
create the environments.

Note:
    The environments only cover phase one of the boss fights so far. For further explanations see
    :mod:`~.envs`.

The second main module is the :mod:`soulsgym.core` module. It contains all necessary
functionalities for the environments to interact with the game. Unless you want to develop your own
environment for :mod:`~.envs` or are trying to contribute, this module can safely be ignored.

``soulsgym`` registers its environments with OpenAI's ``gym`` module on import. In order to use the
environments you follow the usual pattern of OpenAI's ``gym.make``. A list of all available
environments is available at ``soulsgym.available_envs``.
"""
from pathlib import Path
import logging
import sys

from gym.envs.registration import register
if sys.platform == "win32":
    on_windows = True
    import win32api
else:
    on_windows = False

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


if on_windows:
    if not _check_ds3_path():
        logger.warning("Could not find Dark Souls III executable. Continuing for now...")
else:
    logger.info("Running SoulsGym on a non-Windows platform. Most features are not available.")


def set_log_level(level: int):
    """Set log level for the soulsgym module.

    Args:
        level: Logger level for the module. Uses the standard library logging module levels.
    """
    logger.setLevel(level)


available_envs = ["SoulsGymIudex-v0"]

# Register environments in OpenAI gym
register(id="SoulsGymIudex-v0",
         entry_point='soulsgym.envs.iudex_env:IudexEnv',
         nondeterministic=True)
