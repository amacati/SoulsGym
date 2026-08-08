"""Speed hack module for pausing and accelerating the game.

A small DLL is loaded into the game and rewrites its ``Import Address Table`` entries for the Win32
timer functions to scaled versions. Since the game derives its frame delta from these timers,
scaling them slows, accelerates or (at ``speed = 0``) pauses the game. The speed is set at runtime
over a loopback TCP socket, which behaves identically on Windows and under Proton/Wine.

On Windows the DLL is injected on demand via ``CreateRemoteThread`` + ``LoadLibrary``. On
Linux/Proton it is loaded as a ``dinput8.dll`` proxy at game launch via :func:`install_proxy`, which
places the files and sets the ``WINEDLLOVERRIDES="dinput8=n,b"`` launch option. See
:mod:`soulsgym.core.speedhack.speedhack` for details.
"""

from soulsgym.core.speedhack.speedhack import SpeedHackConnector, inject_dll, install_proxy

__all__ = ["inject_dll", "install_proxy", "SpeedHackConnector"]
