"""Operating system detection and the per-platform game boot sequence."""

import platform
import subprocess
import time
from enum import StrEnum

from soulsgym.core.utils import get_pid


class System(StrEnum):
    """Supported operating systems."""

    WINDOWS = "Windows"
    LINUX = "Linux"


SYSTEM = System(platform.system())


def boot(app_id: int, process_name: str):
    """Bring the game process up for the current platform.

    On Windows the game must already be running. On Linux it is launched through Steam if it is not
    running yet.

    Args:
        app_id: The game's Steam app id.
        process_name: The game's process name.
    """
    {System.WINDOWS: _boot_windows, System.LINUX: _boot_linux}[SYSTEM](app_id, process_name)


def _boot_windows(app_id: int, process_name: str):
    assert _running(process_name), "Start the game before booting the interface."


def _boot_linux(app_id: int, process_name: str):
    if _running(process_name):
        return
    # steam stays alive as the client, so launch detached and wait for the game process instead.
    subprocess.Popen(
        ["steam", "-applaunch", str(app_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _wait_running(process_name)


def _running(process_name: str) -> bool:
    try:
        get_pid(process_name)
    except RuntimeError:
        return False
    return True


def _wait_running(process_name: str, timeout: float = 120):
    deadline = time.monotonic() + timeout
    while not _running(process_name):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{process_name} did not start within {timeout} s")
        time.sleep(2)
