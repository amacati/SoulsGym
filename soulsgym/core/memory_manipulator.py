"""The ``memory_manipulator`` module is a wrapper around ``pymem`` for memory read and write access.

It implements some of the basic CheatEngine functionalities in Python. The game is controlled by
changing the values of ingame properties in the process memory. We cannot write to static memory
addresses since the process memory layout is dynamic and changes every time the game loads. Memory
locations are given as chains of pointers instead which we have to resolve to get the current
address for each attribute. These pointer chains were largely copied from available Dark Souls III
cheat tables.

Note:
    Not all game properties of interest were included in the cheat tables. Some values and their
    pointer chains were determined by us and are by no means guaranteed to be stable. Please report
    any memory read or write error to help us identify unstable pointer chains!

Warning:
    We cache resolved pointer chains to increase read and write access times. This requires manual
    cache clearing. For details see :meth:`MemoryManipulator.clear_cache`.

The ``MemoryManipulator`` is writing from an external process to a memory region in use by the game
process. You *will* see race conditions during writing, particularly for values with high frequency
writes in the game loop (e.g. coordinates). Be sure to include checks if writes were successful and
have taken effect in the game when you write to these memory locations.
"""
from __future__ import annotations
from typing import List

import psutil
import win32process
import win32api
import win32con
import pymem as pym
from pymem import Pymem

from soulsgym.core.utils import Singleton
from soulsgym.core.static import address_base_patterns, address_bases


class MemoryManipulator(Singleton):
    """Handle reads and writes to the game process memory.

    The ``MemoryManipulator`` wraps ``pymem`` functions for memory read and writes. It manages the
    game memory pointers, address resolving and decoding.
    """

    def __init__(self, process_name: str = "DarkSoulsIII.exe"):
        """Initialize the cache and pointer attributes.

        If the game is not open, the pointer values can't be inferred which causes an exception.

        Args:
            process_name: The target process name. Should always be DarkSoulsIII.exe, unless the app
                name changes.
        """
        if not hasattr(self, "is_init"):
            self.process_name = process_name
            self.pid = self.get_pid(self.process_name)
            # Get the base address
            self.process_handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, self.pid)
            modules = win32process.EnumProcessModules(self.process_handle)
            self.base_address = modules[0]
            # Create Pymem object once, this has a relative long initialziation
            self.pymem = Pymem()
            self.pymem.open_process_from_id(self.pid)
            self.address_cache = {}

            self.ds_module = pym.process.module_from_name(self.pymem.process_handle,self. process_name)

            self.bases = address_bases.copy()
            for base in address_base_patterns:
                pattern = bytes(address_base_patterns[base]["pattern"], "ASCII")
                addr = pym.pattern.pattern_scan_module(self.pymem.process_handle, self.ds_module, pattern)
                if not addr:
                    raise RuntimeError(f'Pattern "{base} could not be resolved!"')
                if "offset" in address_base_patterns[base]:
                    addr += address_base_patterns[base]["offset"]
                addr = addr + self.pymem.read_long(addr + 3) + 7
                self.bases[base] = addr - self.base_address
                print(base,hex(addr))

    def clear_cache(self):
        """Clear the reference look-up cache of the memory manipulator.

        The ``MemoryManipulator`` caches all pointer chains it resolves to speed up the reads and
        writes. If the game reloads, these addresses are no longer guaranteed to be valid and the
        address cache has to be cleared in order to resolve the new addresses of all values. Cache
        validation by reading the player death count is omitted since it incurs additional overhead
        for read operations and offsets any performance gains made by using an address cache.

        Warning:
            We do not validate the cache before reading from a cached address! It is the users's
            responsibility to clear the cache on reload!
        """
        self.address_cache = {}

    @staticmethod
    def get_pid(process_name: str) -> int:
        """Fetch the process PID of a process identified by a given name.

        Args:
            process_name: The name of the process to get the PID from.

        Returns:
            The process PID.

        Raises:
            RuntimeError: No process with name ``process_name`` currently open.
        """
        for proc in psutil.process_iter():
            if proc.name() == process_name:
                return proc.pid
        raise RuntimeError(f"Process {process_name} not open")

    def resolve_address(self, addr_offsets: List[int], base: int) -> int:
        """Resolve an address by its offsets and a base.

        Looks up the address cache first.

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
        if u_id in self.address_cache:
            return self.address_cache[u_id]
        # When no cache hit: resolve by following the pointer chain until its last link
        helper = self.pymem.read_longlong(base)
        for o in addr_offsets[:-1]:
            helper = self.pymem.read_longlong(helper + o)
        helper += addr_offsets[-1]
        # Add to cache
        self.address_cache[u_id] = helper
        return helper

    def read_int(self, address: int) -> int:
        """Read an integer from memory.

        Args:
            address: The read address.

        Returns:
            The integer value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_long(address)

    def read_float(self, address: int) -> float:
        """Read a float from memory.

        Args:
            address: The read address.

        Returns:
            The float value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_float(address)

    def read_string(self,
                    address: int,
                    length: int,
                    null_term: bool = True,
                    codec: str = "utf-16") -> str:
        """Read a string from memory.

        Args:
            address: The read address.
            length: The expected (maximum) string length.
            null_term: String should be cut after double 0x00.
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

    def read_bytes(self, address: int, length: int) -> bytes:
        """Read raw bytes from memory.

        Args:
            address: The read address.
            length: The bytes length.

        Returns:
            The raw bytes.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_bytes(address, length)

    def write_bit(self, address: int, index: int, value: int):
        """Write a single bit.

        Args:
            address: The write address.
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

    def write_int(self, address: int, value: int):
        """Write an integer to memory.

        Args:
            address: The write address.
            value: The value of the integer.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_long(self.pymem.process_handle, address, value)

    def write_float(self, address: int, value: float):
        """Write a float to memory.

        Args:
            address: The write address.
            value: The value of the float.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_float(self.pymem.process_handle, address, value)

    def write_bytes(self, address: int, buffer: bytes):
        """Write a series of bytes to memory.

        Args:
            address: The write address for the first byte.
            buffer: The bytes.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_bytes(self.pymem.process_handle, address, buffer, len(buffer))
