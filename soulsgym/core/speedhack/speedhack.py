"""Speed hack connector for pausing and accelerating the game.

The speed hack is a small DLL that is loaded into the game and rewires the game's timer functions to
scaled versions. The game derives its frame delta from these, so scaling them slows, accelerates or
pauses the game.

The speed is set at runtime over a loopback TCP socket the DLL listens on. TCP works identically on
native Windows and under Proton/Wine, so this connector is the same on both platforms. Loading the
DLL, however, differs per platform. On Windows, we inject the DLL at runtime via :func:`inject_dll`.
On Linux/Proton, the DLL is loaded as a ``dinput8.dll`` proxy at game launch. We first run
:func:`install_proxy` to place the files. Then, add the steam launch option
``WINEDLLOVERRIDES="dinput8=n,b"``, and start the game. Loading at startup means SoulsGym has to
launch the game on Linux.

The connector is a singleton because the DLL accepts a single controller at a time, while multiple
:class:`.Game` objects may exist.
"""

import logging
import shutil
import socket
import struct
import time
from pathlib import Path

from soulsgym.core.platform import SYSTEM, System

if SYSTEM is System.WINDOWS:  # Windows-only injection imports
    import win32api
    import win32event
    import win32process

from soulsgym.core.utils import Singleton, get_pid
from soulsgym.exception import InjectionFailure

logger = logging.getLogger(__name__)

# Loopback host/port the speed hack DLL listens on must match rust/src/lib.rs.
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 51285

DLL_PATH = Path(__file__).parent / "_bin" / "speedhack.dll"

# Win32 constants for injection.
PROCESS_ALL_ACCESS = 0x000F0000 | 0x00100000 | 0x00000FFF
MEM_CREATE = 0x00001000 | 0x00002000
MEM_RELEASE = 0x00008000
MAX_PATH = 260
PAGE_READWRITE = 0x04


def inject_dll(process_name: str, dll_path: Path):
    """Inject a DLL into the target process.

    Opens a process handle, allocates virtual memory in its address space, writes the DLL path there
    and forces the process to load the DLL.

    Warning:
        Windows only.

    Args:
        process_name: Target process name.
        dll_path: Path to the DLL file.
    """
    if SYSTEM is not System.WINDOWS:
        raise RuntimeError("DLL injection is only supported on Windows")
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
            "Failed to execute virtual allocation for writing the DLL to the targeted process"
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
    win32process.VirtualFreeEx(p_handle, mem_addr, 0, MEM_RELEASE)
    win32api.CloseHandle(p_handle)
    win32api.CloseHandle(t_handle)


def install_proxy(game_dir: Path, proton_dir: Path):
    """Install the speed hack so it loads into the game at launch.

    Places the speed hack and the original library it stands in for into the game directory. The
    game must be closed while this runs. Enabling the loader and starting the game are manual steps
    covered in the setup guide.

    Args:
        game_dir: The game's install directory.
        proton_dir: The Proton installation directory.
    """
    game_dir, proton_dir = Path(game_dir), Path(proton_dir)
    builtin = proton_dir / "dist/lib64/wine/x86_64-windows/dinput8.dll"
    if not builtin.is_file():
        raise FileNotFoundError(f"Proton builtin dinput8 not found at {builtin}")
    shutil.copy(DLL_PATH, game_dir / "dinput8.dll")
    shutil.copy(builtin, game_dir / "dinput8_hook.dll")
    logger.debug(f"Speed hack proxy installed in {game_dir}.")


def _check_status(sock: socket.socket):
    """Read the speed hack's status byte and raise if it reports a failure."""
    data = sock.recv(1)
    if len(data) != 1:
        raise InjectionFailure("Speed hack connected but did not report its status")
    if data[0]:
        raise InjectionFailure("Speed hack loaded but failed to hook one or more game timers.")


class SpeedHackConnector(metaclass=Singleton):
    """Set the game speed at runtime through the speed hack.

    A single connector controls the speed, shared across all game objects. On Windows the speed
    hack is injected on demand. On Linux it must already be loaded into the running game.
    """

    dll_path = DLL_PATH

    def __init__(self, process_name: str = "DarkSoulsIII.exe"):
        """Connect to the speed hack, injecting it first if necessary.

        Args:
            process_name: Name of the game process.

        Raises:
            InjectionFailure: The speed hack could not be reached.
        """
        self.process_name = process_name
        self.sock: socket.socket | None = None
        try:
            self.sock = self._connect()
            logger.info("Speed hack already active, connected to control port")
            return
        except OSError:
            pass  # Not loaded yet
        if SYSTEM is System.WINDOWS:
            logger.info("Speed hack not active, injecting DLL")
            inject_dll(process_name, self.dll_path)
            self.sock = self._connect(retries=50, delay=0.1)
            logger.info("Speed hack injection successful")
        else:
            raise InjectionFailure(
                "Speed hack control port is not open. The DLL must be loaded as a dinput8 proxy at "
                'game launch. Ensure the launch option WINEDLLOVERRIDES="dinput8=n,b" is used, then '
                "start the game."
            )

    def update_game_speed(self, value: float):
        """Update the game speed to a new value.

        Args:
            value: The new game speed. Must be >= 0. 0 pauses the game.
        """
        assert value >= 0, "Game speed must be >= 0"
        self.sock.sendall(struct.pack("<f", value))

    def _connect(self, retries: int = 1, delay: float = 0.1) -> socket.socket:
        """Open a TCP connection to the DLL's control port, retrying while it comes up.

        Args:
            retries: Number of connection attempts.
            delay: Delay between attempts in seconds.

        Returns:
            The connected socket.

        Raises:
            OSError: The control port could not be reached within the retry budget.
            InjectionFailure: The DLL never opened the control port, or reported a failure status.
        """
        for i in range(retries):
            try:
                sock = socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=2)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _check_status(sock)
                return sock
            except OSError as e:
                if i == retries - 1:
                    raise e
                time.sleep(delay)
        raise InjectionFailure("Speed hack loaded but control port never opened")

    def __del__(self):
        """Close the control socket on deletion."""
        if self.sock is not None:
            self.sock.close()
