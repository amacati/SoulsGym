"""The injection functions inject a DLL into a target process and force it to load the library.

First, we obtain a handle of the target process and allocate memory in the external process. Then we
write the path to the DLL into the allocated memory, create a remote thread in the target process,
and call ``LoadLibrary`` on ``kernel32.lib`` to force the process to load the DLL. The allocated
memory is subsequently released and the handles are closed. This completes the injection.

The injection reroutes the game's calls to Windows' performance timer functions to custom timers.
These timers have an adjustable speed parameter to slow and/or accelerate the game. In order to
dynamically adjust the speed, a `NamedPipe` is opened and continuously read by a second process. The
values sent over the pipe are then used to update the speed parameter at runtime.

The module implements a singleton that provides thread-safe access to the named pipe created by the
DLL. This is necessary because only a limited number of clients can connect to the pipe. Otherwise,
multiple game interface objects could not be instantiated.
"""

import logging
import platform
import struct
import time
from multiprocessing import Lock
from pathlib import Path

if platform.system() == "Windows":  # Windows imports, ignore for unix to make imports work
    import pywintypes
    import win32api
    import win32event
    import win32file
    import win32process

from soulsgym.core.utils import Singleton, get_pid
from soulsgym.exception import InjectionFailure

logger = logging.getLogger(__name__)

PROCESS_ALL_ACCESS = 0x000F0000 | 0x00100000 | 0x00000FFF
MEM_CREATE = 0x00001000 | 0x00002000
MEM_RELEASE = 0x00008000
MAX_PATH = 260
PAGE_READWRITE = 0x04


def inject_dll(process_name: str, dll_path: Path):
    """Inject a DLL into the target process.

    Injection opens a process handle of the target process, allocates virtual memory in its address
    space, writes the DLL path to this address and forces the process to open the DLL by creating a
    remote thread.

    Args:
        process_name: Target process name.
        dll_path: Path to the DLL file.
    """
    try:
        pid = get_pid(process_name)
    except RuntimeError:
        raise InjectionFailure(f"Process {process_name} not found.")
    p_handle = win32api.OpenProcess(PROCESS_ALL_ACCESS, 0, pid)
    mem_addr = _write_dll_to_process(p_handle, dll_path)
    t_handle = _create_remote_thread(p_handle, mem_addr)
    win32event.WaitForSingleObject(t_handle, 5_000)
    _injection_cleanup(p_handle, t_handle, mem_addr)


def _write_dll_to_process(p_handle: int, dll_path: Path) -> int:
    """Write the DLL path to the external process memory.

    Args:
        p_handle: Target process handle.
        dll_path: Path to the injection DLL.

    Returns:
        The memory address of the allocated memory.

    Raises:
        InjectionFailure: The requested memory could not be allocated or written to.
    """
    mem_addr = win32process.VirtualAllocEx(p_handle, 0, MAX_PATH, MEM_CREATE, PAGE_READWRITE)
    if not mem_addr:
        win32api.CloseHandle(p_handle)
        raise InjectionFailure(
            ("Failed to execute virtual allocation for writing the DLL to the" " targeted process")
        )
    if not win32process.WriteProcessMemory(p_handle, mem_addr, str(dll_path).encode("ascii")):
        win32api.CloseHandle(p_handle)
        raise InjectionFailure("Failed to write DLL path to allocated memory in targeted process")
    return mem_addr


def _create_remote_thread(p_handle: int, mem_addr: int) -> int:
    """Create a remote thread and load the injected DLL.

    Args:
        p_handle: Target process handle.
        mem_addr: Address of the DLL path memory.

    Returns:
        The handle of the created thread.
    """
    module_addr = win32api.GetModuleHandle("kernel32.dll")
    load_library_fn = win32api.GetProcAddress(module_addr, "LoadLibraryA")
    t_handle, _ = win32process.CreateRemoteThread(p_handle, None, 0, load_library_fn, mem_addr, 0)
    return t_handle


def _injection_cleanup(p_handle: int, t_handle: int, mem_addr: int):
    """Clean up the injection by freeing the allocated memory and closing the handles.

    Args:
        p_handle: Target process handle.
        t_handle: Injection thread handle.
        mem_addr: Address of the DLL path memory.
    """
    win32process.VirtualFreeEx(p_handle, mem_addr, 0, MEM_RELEASE)  # TODO: Enable Mem Free
    win32api.CloseHandle(p_handle)
    win32api.CloseHandle(t_handle)


class SpeedHackConnector(metaclass=Singleton):
    """Inject the speed hack and connect to the speed command pipe.

    The connector is designed as singleton as only a single connection to the pipe is allowed, but
    multiple :class:`.Game` objects might exist. The connector provides a thread-safe method to
    communicate with the injected pipe using only a single client.

    Note:
        The ``SpeedHackDLL.dll`` is compiled to `soulsgym/core/speedhack/_C/x64/Release`. In order
        to use it for the injection, either update the path or move the dll file into the `_C`
        folder.
    """

    pipe_name = r"\\.\pipe\SoulsGymSpeedHackPipe"
    dll_path = Path(__file__).parent / "_C" / "SpeedHackDLL.dll"
    _lock = Lock()

    def __init__(self, process_name: str):
        """Connect to the SpeedHack pipe.

        If the pipe is not yet open, inject the DLL into the game.

        Args:
            process_name: Name of the process to connect to.
        """
        self.pipe = None
        self.target_name = process_name
        try:
            self.pipe = self._connect_pipe()
            logger.info("SpeedHack already enabled, skipping injection")
        except pywintypes.error as e:  # Pipe may not be open. In that case, we have to inject first
            if not e.args[0] == 2 and e.args[1] == "CreateFile":
                raise e  # Not the anticipated error on missing pipe, so we re-raise
            logger.info("SpeedHack not active, proceeding with injection")
            inject_dll(self.target_name, self.dll_path)
            time.sleep(0.1)  # Give the pipe time to come up after the injection (very conservative)
            self.pipe = self._connect_pipe()
            logger.info("SpeedHack activation successful")

    def set_game_speed(self, value: float):
        """Set the game speed to a new value.

        Args:
            value: The new game speed. Can't be lower than 0.
        """
        assert value >= 0
        win32file.WriteFile(self.pipe, struct.pack("f", value))

    def _connect_pipe(self) -> int:
        return win32file.CreateFile(
            self.pipe_name, win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None
        )

    def __del__(self):
        """Close the pipe handle on deletion."""
        if self.pipe is not None:
            try:
                win32file.CloseHandle(self.pipe)
            except TypeError:  # Throws NoneType object not callable error for some reason
                ...
