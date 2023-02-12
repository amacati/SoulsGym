"""Speed hack module for pausing and accelerating the game.

The injection functions inject a DLL into a target process and force it to load the library. First,
we obtain a handle of the target process and allocate memory in the external process. Then we write
the path to the DLL into the allocated memory, create a remote thread in the target process, and
call ``LoadLibrary`` on ``kernel32.lib`` to force the process to load the DLL. The allocated memory
is subsequently released and the handles are closed. This completes the injection.

The injection reroutes the game's calls to Windows' performance timer functions to custom timers.
These timers have an adjustable speed parameter to slow and/or accelerate the game. In order to
dynamically adjust the speed, a `NamedPipe` is opened and continuously read by a second process. The
values sent over the pipe are then used to update the speed parameter at runtime.

The module implements a singleton that provides thread-safe access to the named pipe created by the
DLL. This is necessary because only a limited number of clients can connect to the pipe. Otherwise,
multiple game interface objects could not be instantiated.
"""
from soulsgym.core.speedhack.speedhack import inject_dll, SpeedHackConnector

__all__ = ["inject_dll", "SpeedHackConnector"]
