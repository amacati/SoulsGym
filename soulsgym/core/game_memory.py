"""The ``game_memory`` module reads and writes the game's process memory.

It implements some of the basic CheatEngine functionalities in Python. The game is controlled by
changing the values of ingame properties in the process memory. We cannot write to static memory
addresses since the process memory layout is dynamic and changes every time the game loads. Memory
locations are given as chains of pointers instead which we have to resolve to get the current
address for each attribute. These pointer chains were largely copied from available Dark Souls III
cheat tables.

The low-level reads and writes go through a platform-specific backend that accesses the game process
from the outside, built on ``pymem`` on Windows and on ``glibc`` on Linux.

Note:
    Not all game properties of interest were included in the cheat tables. Some values and their
    pointer chains were determined by us and are by no means guaranteed to be stable. Please report
    any memory read or write error to help us identify unstable pointer chains!

Warning:
    We cache resolved pointer chains to increase read and write access times. This requires manual
    cache clearing. For details see :meth:`GameMemory.clear_cache`.

The ``GameMemory`` is writing from an external process to a memory region in use by the game
process. You *will* see race conditions during writing, particularly for values with high frequency
writes in the game loop (e.g. coordinates). Be sure to include checks if writes were successful and
have taken effect in the game when you write to these memory locations.
"""

from __future__ import annotations

import ctypes
import re
import struct
from typing import NotRequired, TypedDict

from iced_x86 import Decoder

from soulsgym.core.platform import SYSTEM, System
from soulsgym.core.static import address_base_patterns, address_bases
from soulsgym.core.utils import Singleton, get_pid
from soulsgym.exception import MemoryPatternScanError, MemoryReadError, MemoryWriteError

if SYSTEM is System.WINDOWS:  # Windows-only backend imports
    import pymem as pym
    import win32api
    import win32con
    import win32process
    from pymem import Pymem
    from pymem.exception import MemoryReadError as PymemReadError
    from pymem.exception import MemoryWriteError as PymemWriteError


class AddressRecord(TypedDict):
    """Type definition for an address record."""

    base: str
    offsets: list[int]
    type: str
    length: NotRequired[int]  # Length of string or byte arrays


class GameMemory(metaclass=Singleton):
    """Handle reads and writes to the game process memory.

    The ``GameMemory`` manages the game memory pointers, address resolving and decoding on top of a
    platform-specific low-level backend.
    """

    def __init__(self, process_name: str = "DarkSoulsIII.exe"):
        """Initialize the cache and pointer attributes.

        If the game is not open, the pointer values can't be inferred which causes an exception.

        Args:
            process_name: The target process name. Should always be DarkSoulsIII.exe, unless the app
                name changes.
        """
        self.process_name = process_name
        self.pid = get_pid(self.process_name)
        if SYSTEM is System.WINDOWS:
            self._backend = _PymemBackend(self.pid, process_name)
        elif SYSTEM is System.LINUX:
            self._backend = _GLibCBackend(self.pid, process_name)
        else:
            raise NotImplementedError(f"GameMemory not implemented for {SYSTEM}!")
        self.base_address = self._backend.base_address
        self.address_cache: dict[str, int] = {}
        self.bases = self._load_bases(process_name)

    def resolve_record(self, record: AddressRecord) -> int:
        """Resolve an address record by following its pointer chain to the final address.

        Resolved addresses are cached to increase performance. If the program reallocates memory,
        the cached addresses are no longer valid and the cache has to be cleared.

        Warning:
            Can't detect an invalid cache, this is the user's responsibility!

        Args:
            record: The address record. Address records must contain at least the `base` and
                `offsets` keys.

        Returns:
            The resolved address.
        """
        unique_address_id = str((record["offsets"], record["base"]))
        if unique_address_id in self.address_cache:  # Look up the cache first
            return self.address_cache[unique_address_id]
        # When no cache hit: resolve by following the pointer chain until its last link
        address = self._read_ptr(self.bases[record["base"]])
        for offset in record["offsets"][:-1]:
            address = self._read_ptr(address + offset)
        address += record["offsets"][-1]
        self.address_cache[unique_address_id] = address  # Add resolved address to cache
        return address

    def clear_cache(self):
        """Clear the reference look-up cache of the game memory.

        The ``GameMemory`` caches all pointer chains it resolves to speed up the reads and writes.
        If the game reloads, these addresses are no longer guaranteed to be valid and the address
        cache has to be cleared in order to resolve the new addresses of all values. Cache
        validation by reading the player death count is omitted since it incurs additional overhead
        for read operations and offsets any performance gains made by using an address cache.

        Warning:
            We do not validate the cache before reading from a cached address! It is the users's
            responsibility to clear the cache on reload!
        """
        self.address_cache = {}

    def read_record(self, record: AddressRecord) -> int | float | str | bytes:
        """Resolve the record address and read the value into the hinted type.

        Args:
            record: The address record.

        Returns:
            The read value.
        """
        address = self.resolve_record(record)
        match record["type"]:
            case "int":
                return self.read_int(address)
            case "float":
                return self.read_float(address)
            case "str":
                return self.read_string(address, length=record["length"])
            case "bytes":
                return self.read_bytes(address, record["length"])
            case _:
                raise ValueError(f"Type '{record['type']}' not supported!")

    def write_record(self, record: AddressRecord, value: int | float | bytes):
        """Resolve the record address and write the value to the address.

        The provided value has to match the type hint of the record.

        Note:
            We support reading string records, but not writing them.

        Args:
            record: The address record.
            value: The value to write. Type has to match the type hint of the record.
        """
        address = self.resolve_record(record)
        match record["type"]:
            case "int":
                assert isinstance(value, int), f"Trying to write {type(value)} to int record!"
                self.write_int(address, value)
            case "float":
                assert isinstance(value, (int, float)), (
                    f"Trying to write {type(value)} to float record!"
                )
                self.write_float(address, value)
            case "bytes":
                assert isinstance(value, bytes), f"Trying to write {type(value)} to byte record!"
                self.write_bytes(address, value)
            case _:
                raise ValueError(f"Type '{record['type']}' not supported!")

    def read_int(self, address: int) -> int:
        """Read an integer from memory.

        Args:
            address: The read address.

        Returns:
            The integer value.

        Raises:
            MemoryReadError: An error with the memory read occured.
        """
        return int.from_bytes(self._backend.read_bytes(address, 4), "little", signed=True)

    def read_float(self, address: int) -> float:
        """Read a float from memory.

        Args:
            address: The read address.

        Returns:
            The float value.

        Raises:
            MemoryReadError: An error with the memory read occured.
        """
        return struct.unpack("<f", self._backend.read_bytes(address, 4))[0]

    def read_string(self, address: int, length: int, null_term: bool = True) -> str:
        """Read a UTF-16 string from memory.

        Args:
            address: The read address.
            length: The expected (maximum) string length in bytes.
            null_term: Cut the string at the first UTF-16 null terminator.

        Returns:
            The string.

        Raises:
            MemoryReadError: An error with the memory read occured.
            UnicodeDecodeError: An error with the decoding of the read bytes occured.
        """
        s = self._backend.read_bytes(address, length)
        if null_term:
            i = s.find(b"\x00\x00")
            while i != -1 and i % 2:  # odd offsets are not utf-16 code-unit boundaries
                i = s.find(b"\x00\x00", i + 1)
            if i != -1:
                s = s[:i]
        return s.decode("utf-16")

    def read_bytes(self, address: int, length: int) -> bytes:
        """Read raw bytes from memory.

        Args:
            address: The read address.
            length: The bytes length.

        Returns:
            The raw bytes.

        Raises:
            MemoryReadError: An error with the memory read occured.
        """
        return self._backend.read_bytes(address, length)

    def write_bit(self, address: int, index: int, value: int):
        """Write a single bit.

        Args:
            address: The write address.
            index: The index of the bit (0 ... 7).
            value: The value of the bit (0/1).

        Raises:
            MemoryWriteError: An error with the memory write occured.
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
            MemoryWriteError: An error with the memory write occured.
        """
        self.write_bytes(address, struct.pack("<i", value))

    def write_float(self, address: int, value: float):
        """Write a float to memory.

        Args:
            address: The write address.
            value: The value of the float.

        Raises:
            MemoryWriteError: An error with the memory write occured.
        """
        self.write_bytes(address, struct.pack("<f", value))

    def write_bytes(self, address: int, buffer: bytes):
        """Write a series of bytes to memory.

        Args:
            address: The write address for the first byte.
            buffer: The bytes.

        Raises:
            MemoryWriteError: An error with the memory write occured.
        """
        self._backend.write_bytes(address, buffer)

    def _read_ptr(self, address: int) -> int:
        """Read an 8-byte pointer from memory."""
        return int.from_bytes(self._backend.read_bytes(address, 8), "little")

    def _load_bases(self, process_name: str) -> dict:
        match process_name:
            case "DarkSoulsIII.exe":
                game = "DarkSoulsIII"
            case "eldenring.exe":  # Not an error, eldenring.exe isn't capitalized
                game = "EldenRing"
            case _:
                raise ValueError(f"Process name '{process_name}' not supported!")
        if address_bases[game] is None:
            bases = {}
        else:
            bases = {name: addr + self.base_address for name, addr in address_bases[game].items()}
        for base_key, base in address_base_patterns[game].items():
            pattern = bytes(base["pattern"], "ASCII")
            addr = self._backend.pattern_scan(pattern)
            if "offset" in base:
                addr += base["offset"]
            # The pattern matches a rip-relative instruction that references the base. Disassembling
            # it and resolving the operand gives us the base of the pointer chain. For a chain
            # [["DarkSoulsIII.exe" + 0x123] + 0x456] we store "DarkSoulsIII.exe" + 0x123.
            # Basically the same as the CheatEngine disassembler does to get the base addresses
            code = self._backend.read_bytes(addr, 15)  # a single instruction, at most 15 bytes
            instr = Decoder(64, code, ip=addr).decode()
            assert instr.is_ip_rel_memory_operand, f"{base_key} pattern is not rip-relative"
            bases[base_key] = instr.ip_rel_memory_address
        return bases


class _PymemBackend:
    """Windows memory backend built on pymem."""

    def __init__(self, pid: int, process_name: str):
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
        self.base_address = win32process.EnumProcessModules(handle)[0]
        self.pymem = Pymem()
        self.pymem.open_process_from_id(pid)
        self._module = pym.process.module_from_name(self.pymem.process_handle, process_name)

    def read_bytes(self, address: int, length: int) -> bytes:
        try:
            return self.pymem.read_bytes(address, length)
        except PymemReadError as e:
            raise MemoryReadError(address, length) from e

    def write_bytes(self, address: int, data: bytes):
        try:
            pym.memory.write_bytes(self.pymem.process_handle, address, data, len(data))
        except PymemWriteError as e:
            raise MemoryWriteError(address, data) from e

    def pattern_scan(self, pattern: bytes) -> int:
        addr = pym.pattern.pattern_scan_module(self.pymem.process_handle, self._module, pattern)
        if addr is None:
            raise MemoryPatternScanError("Could not find pattern in module memory!")
        return addr


class _iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


class _GLibCBackend:
    """Linux memory backend using glibc.

    The game runs as a Windows executable inside a Wine process and keeps its virtual address
    layout, so the same base and offset pointer chains resolve as on Windows.
    """

    def __init__(self, pid: int, process_name: str):
        self.pid = pid
        libc = ctypes.CDLL(None, use_errno=True)  # Fetch the glibc symbols from the current process
        # ctypes expects 32bit values, so we first configure it for compatibility with 64bit inputs.
        # Likely doesn't matter for our use case, but ensures correctness
        argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_iovec),
            ctypes.c_ulong,
            ctypes.POINTER(_iovec),
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.process_vm_readv.restype = ctypes.c_ssize_t
        libc.process_vm_readv.argtypes = argtypes
        libc.process_vm_writev.restype = ctypes.c_ssize_t
        libc.process_vm_writev.argtypes = argtypes
        self._libc = libc
        self.base_address = self._module_base(process_name)
        self._image: list[tuple[int, bytes]] | None = None  # lazy snapshot for pattern scanning

    def _module_base(self, process_name: str) -> int:
        with open(f"/proc/{self.pid}/maps") as f:
            for line in f:  # Lines are sorted by address, so the first match should be the base
                if line.rstrip().endswith(process_name):
                    return int(line.split("-", 1)[0], 16)
        raise RuntimeError(f"Module {process_name} not found in /proc/{self.pid}/maps")

    def read_bytes(self, address: int, length: int) -> bytes:
        buf = (ctypes.c_char * length)()
        local = _iovec(ctypes.cast(buf, ctypes.c_void_p), length)
        remote = _iovec(ctypes.c_void_p(address), length)
        n = self._libc.process_vm_readv(
            self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0
        )
        if n != length:
            raise MemoryReadError(address, length, ctypes.get_errno())
        return buf.raw

    def write_bytes(self, address: int, data: bytes):
        buf = ctypes.create_string_buffer(data, len(data))
        local = _iovec(ctypes.cast(buf, ctypes.c_void_p), len(data))
        remote = _iovec(ctypes.c_void_p(address), len(data))
        n = self._libc.process_vm_writev(
            self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0
        )
        if n != len(data):
            raise MemoryWriteError(address, data, ctypes.get_errno())

    def pattern_scan(self, pattern: bytes) -> int:
        if self._image is None:
            self._image = self._snapshot_image()
        regex = re.compile(pattern, re.DOTALL)
        for start, data in self._image:
            match = regex.search(data)
            if match:
                return start + match.start()
        raise MemoryPatternScanError("Could not find pattern in module memory!")

    def _snapshot_image(self) -> list[tuple[int, bytes]]:
        """Read all readable memory regions belonging to the game module (for AOB scanning)."""
        base = self.base_address
        # The DOS stub holds the PE header offset at 0x3C
        # See https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#ms-dos-stub-image-only
        pe_header_offset = int.from_bytes(self.read_bytes(base + 0x3C, 4), "little")
        # The optional header starts at pe_header_offset + 24. 4 bytes come from the PE signature,
        # 20 from the COFF header. In the PE32+ optional header, SizeOfImage has an offset of 56.
        # See https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-windows-specific-fields-image-only
        size = int.from_bytes(self.read_bytes(base + pe_header_offset + 24 + 56, 4), "little")
        end = base + size
        regions: list[tuple[int, bytes]] = []
        with open(f"/proc/{self.pid}/maps") as f:
            for line in f:
                addrs, perms, *_ = line.split()
                if "r" not in perms:  # We only want readable regions, skip the rest
                    continue
                start, stop = (int(x, 16) for x in addrs.split("-"))
                if stop <= base or start >= end:  # Out of module bounds, skip
                    continue
                start, stop = max(start, base), min(stop, end)
                try:
                    regions.append((start, self.read_bytes(start, stop - start)))
                except MemoryReadError:
                    continue  # region not readable via process_vm_readv, skip
        return regions
