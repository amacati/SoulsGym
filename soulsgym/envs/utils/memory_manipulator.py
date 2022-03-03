"""Low level memory manipulation interface.

Since memory access has to be fast, we instantiate an object of the _MemoryManipulator class at
import time and export the MemoryManipulator object itself.
"""
from __future__ import annotations
import ctypes
from typing import Any, Callable, List, Optional

import psutil
import win32process
import win32api
import win32con
import pymem as pym
from pymem import Pymem

BASES = {
    "A": 0x4740178,
    "B": 0x4768E78,
    "C": 0x4743AB0,
    "D": 0x4743A80,
    "GameFlagData": 0x473BE28,
    "GlobalSpeed": 0x999C28,
}

VALUE_ADDRESS_OFFSETS = {
    "PlayerHP": [0x80, 0x1F90, 0x18, 0xD8],
    "PlayerMaxHP": [0x80, 0x1F90, 0x18, 0xDC],
    "PlayerSP": [0x80, 0x1F90, 0x18, 0xF0],
    "PlayerMaxSP": [0x80, 0x1F90, 0x18, 0xF4],
    "IudexDefeated": [0x0, 0x5A67],  # Bit 7 saves the defeat flag!
    "TargetedHP": [0x1F90, 0x18, 0xD8],
    "TargetedMaxHP": [0x1F90, 0x18, 0xE0],
    "PlayerX": [0x40, 0x28, 0x80],
    "PlayerY": [0x40, 0x28, 0x88],
    "PlayerZ": [0x40, 0x28, 0x84],
    "PlayerA": [0x40, 0x28, 0x74],
    "PlayerAnimation": [0x80, 0x1F90, 0x28, 0x898],
    "TargetedX": [0x1F90, 0x68, 0xa8, 0x40, 0x70],
    "TargetedY": [0x1F90, 0x68, 0xa8, 0x40, 0x78],
    "TargetedZ": [0x1F90, 0x68, 0xa8, 0x40, 0x74],
    "TargetedA": [0x1F90, 0x68, 0xa8, 0x40, 0x7C],
    "TargetedAnimation": [0x1F90, 0x28, 0x898],
    "PlayerSpeedMod": [0x80, 0x1F90, 0x28, 0xA58],
    "LockedOnFlag": [0x70],
    "CamQ1": [0x18, 0xE8, 0x28],
    "CamQ2": [0x18, 0xE8, 0x30],
    "CamQ3": [0x18, 0xE8, 0x34],
    "CamQ4": [0x18, 0xE8, 0x38],
    "CameraX": [0x18, 0xE8, 0x40],
    "CameraY": [0x18, 0xE8, 0x48],
    "CameraZ": [0x18, 0xE8, 0x44],
    "noGravity": [0x80, 0x1a08]  # Bit 6 saves the gravity flag!
}

TARGETED_ENTITY_CODE_POS = 0x85a74a
TARGETED_ENTITY_OLD_CODE = b"\x48\x8b\x80\x90\x1f\x00\x00"

# Docs: https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-VIRTUAL_ALLOC
VIRTUAL_ALLOC = ctypes.windll.kernel32.VirtualAlloc
VIRTUAL_ALLOC.argtypes = [
    ctypes.wintypes.LPVOID, ctypes.c_ulonglong, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD
]
VIRTUAL_ALLOC.restype = ctypes.c_ulonglong

# currently not needed
# VirtualFree = ctypes.windll.kernel32.VirtualFree
# VirtualFree.restype = ctypes.wintypes.BOOL
# VirtualFree.argtypes = [ ctypes.wintypes.LPVOID, ctypes.c_ulonglong, ctypes.wintypes.DWORD ]


class _MemoryManipulator:
    """Handle the memory manipulation of the game.

    At heart this class wraps pymem functions for memory read and writes. For better useability it
    manages the game memory pointers, address resolving and decoding.
    """

    def __init__(self, process_name: str = "DarkSoulsIII.exe"):
        """Initialize the cache and pointer attributes.

        If the game is not open, the pointer values can't be inferred which causes an exception.
        However we immediately initialize this class on file import. In order to make an import
        possible without starting Dark Souls III first, we mark functions that need valid pointer
        values and lazily retry initializing the pointers on first actual use.

        Args:
            process_name: The target process name. Should always be DarkSoulsIII.exe, unless the app
                name changes.
        """
        self.cache = {}
        self.process_name = process_name
        # Initialize attributes of lazy_initialize in __init__
        self.pid = None
        self.process_handle = None
        self.base_address = None
        self.pymem = None
        self.target_ptr = None
        self.target_event = None
        self.target_ptr_volatile = None
        try:
            self.lazy_initalize()
            self.init = True
            self._inject_targeted_entity_tracking()
        except Exception as e:
            print(e)
            self.init = False

    def lazy_initalize(self):
        """Initialize the relevant pointers into the Dark Souls III game memory."""
        self.pid = self.get_pid(self.process_name)
        # Get the base address
        self.process_handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, self.pid)
        modules = win32process.EnumProcessModules(self.process_handle)
        self.base_address = modules[0]

        # Create Pymem object once, this has a relative long initialziation
        self.pymem = Pymem()
        self.pymem.open_process_from_id(self.pid)
        self.target_ptr = self.pymem.allocate(8)
        self.target_event = self.pymem.allocate(8)
        self.target_ptr_volatile = self.pymem.allocate(8)

    def require_initalize(func) -> Callable:
        """Decorate functions that require properly initialized memory pointers.

        Args:
            func: The target function for the decorator.

        Returns:
            A wrapper which initializes the pointers if they are still invalid and then executes the
            target function.
        """

        def wrapper(self: _MemoryManipulator, *args: Any, **kwargs: Any) -> Any:
            if not self.init:
                self.lazy_initalize()
                self.init = True
                self._inject_targeted_entity_tracking()
            return func(self, *args, **kwargs)

        return wrapper

    def clear_cache(self):
        """Clear the reference look-up cache of the memory manipulator."""
        self.cache = {}

    @staticmethod
    def get_pid(process_name: str) -> Optional[int]:
        """Fetch the process PID of a process identified by a given name.

        Args:
            process_name: The name of the process to get the PID from.

        Returns:
            The process PID.
        """
        for proc in psutil.process_iter():
            if proc.name() == process_name:
                return proc.pid

    @require_initalize
    def resolve_address(self, addr_offsets: List[int], base: int) -> int:
        """Resolve an address by its offsets and a base.

        Looks up the cache first.

        Warning:
            Can't detect an invalid cache, this is the user's responsibility!

        Args:
            addr_offsets: The offsets which will be resolved iteratively. The first offset is the
            offset to the base itself.
            base: The base offset from the start of the program's memory.

        Returns:
            The resolved address.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        u_id = str((addr_offsets, base))
        # Look up the cache
        if u_id in self.cache:
            return self.cache[u_id]
        # When no cache hit: resolve
        helper = self.pymem.read_longlong(base)
        for o in addr_offsets[:-1]:
            helper = self.pymem.read_longlong(helper + o)
        helper += addr_offsets[-1]
        # Add to cache
        self.cache[u_id] = helper
        return helper

    @require_initalize
    def read_int(self, address: int) -> int:
        """Read an integer from memory.

        Args:
            address: The address to be looked into.

        Returns:
            The integer value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_long(address)

    @require_initalize
    def read_float(self, address: int) -> float:
        """Read a float from memory.

        Args:
            address: The address to be looked into.

        Returns:
            The float value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_float(address)

    @require_initalize
    def read_string(self,
                    address: int,
                    length: int,
                    null_term: bool = True,
                    codec: str = "utf-16") -> str:
        """Read a string from memory.

        Args:
            address: The address to be looked into.
            length: The expected (maximum) string length.
            null_term: Whether the string shall be cut after double 0x00.
            codec: The codec used to decode the bytes.

        Returns:
            The string.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
            UnicodeDecodeError: An error with the decoding of the read bytes occured.
        """
        s = self.pymem.read_bytes(address, length)
        if null_term:
            pos = 0
            for i in range(1, length, 2):
                if s[i - 1] == 0x00 and s[i] == 0x00:
                    pos = i
                    break
            s = s[:pos - 1]
            if not pos:
                s = s + bytes(1)  # Add null termination for strings which exceed 20 chars.
        return s.decode(codec)

    @require_initalize
    def read_bytes(self, address: int, length: int) -> bytes:
        """Read raw bytes from memory.

        Args:
            address: The address to be looked into.
            length: The amount of bytes that should be read.

        Returns:
            The raw bytes.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_bytes(address, length)

    @require_initalize
    def write_bit(self, address: int, index: int, value: int):
        """Write a single bit.

        Args:
            address: The address to be written into.
            index: The index of the bit (0 ... 7).
            value: The value of the bit (0/1).

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        byte = self.read_bytes(address, 1)
        mask = (1 << index).to_bytes(1, "little")
        byte = (byte[0] & ~mask[0]).to_bytes(1, "little")
        if value:
            byte = (byte[0] | mask[0]).to_bytes(1, "little")
        self.write_bytes(address, byte)

    @require_initalize
    def write_int(self, address: int, value: int):
        """Write an integer to memory.

        Args:
            address: The address to be written into.
            value: The value of the integer.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_long(self.pymem.process_handle, address, value)

    @require_initalize
    def write_float(self, address: int, value: float):
        """Write a float to memory.

        Args:
            address: The address to be written into.
            value: The value of the float.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_float(self.pymem.process_handle, address, value)

    @require_initalize
    def write_bytes(self, address: int, buffer: bytes):
        """Write a series of bytes to memory.

        Args:
            address: The first address to be written into.
            buffer: The bytes to write.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_bytes(self.pymem.process_handle, address, buffer, len(buffer))

    @require_initalize
    def activate_targeted_entity_info(self):
        """Inject the target entity info code pointer into targeted entity info.

        Writes the pointer into a JMP instruction at targeted entity info.
        """
        self.write_bytes(self.base_address + TARGETED_ENTITY_CODE_POS,
                         self.targeted_entity_injection)

    @require_initalize
    def deactivate_targeted_entity_info(self):
        """Inject the default game code pointer into the JMP instruction at targeted entity info."""
        self.write_bytes(self.target_ptr_volatile, bytes(8))  # reset volatile ptr
        self.write_bytes(self.base_address + TARGETED_ENTITY_CODE_POS, TARGETED_ENTITY_OLD_CODE)

    @require_initalize
    def _inject_targeted_entity_tracking(self):
        """Inject the targeted entity tracking binary into the running game."""
        # Make space for injection.
        # TODO: We need to fix this later, like wtf. Currently allocates at an address within the
        # process memory which we found to not cause the game to immediately crash and overwrites
        # the memory with our code
        inj_point = VIRTUAL_ALLOC(self.base_address + 0x10000, 80,
                                  win32con.MEM_RESERVE | win32con.MEM_COMMIT,
                                  win32con.PAGE_READWRITE)
        # JMP has 1 + 4 offset bytes.
        adr_offset = inj_point - (self.base_address + TARGETED_ENTITY_CODE_POS + 5)
        # This byte series is assembler code which allows us to set up the targeted entity tracking
        code = b'\x48\x8b\x48\x58\x48\x8b\x89\x20\x03\x00\x00\x48\x81\xc1\x20\x74\x00\x00\x51\x48\xb9'  # noqa: E501
        code += self.target_event.to_bytes(8, byteorder="little", signed=False)
        code += b'\x48\x8f\x01\x48\xa3'
        code += self.target_ptr.to_bytes(8, byteorder="little", signed=False)
        code += b'\x48\xa3'
        code += self.target_ptr_volatile.to_bytes(8, byteorder="little", signed=False)
        code += b'\x48\x8b\x80\x90\x1f\x00\x00\xe9'
        # Offset Calculation: target: base_address + code + 7 (jmp and nop are one byte each)
        # Current: allocated_address + all_bytes_written + 4 (jump offset)
        jmp_adr = (self.base_address + TARGETED_ENTITY_CODE_POS + 7) - (inj_point + len(code) + 4)
        code += jmp_adr.to_bytes(4, byteorder="little", signed=True)  # Add jump address
        injection = b'\xe9' + adr_offset.to_bytes(4, byteorder="little", signed=True) + b'\x90\x90'
        self.targeted_entity_injection = injection
        pym.memory.write_bytes(self.pymem.process_handle, inj_point, code, len(code))


# MemoryManipulator is an already instanced class of type _MemoryManipulator!
# _MemoryManipulator instancing takes too long for single function calls, therefore we offer an
# already instanced object for functions to import

MemoryManipulator = _MemoryManipulator()
