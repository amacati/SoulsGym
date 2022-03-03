import ctypes
import psutil
import win32process
import win32api
import pymem as pym
from pymem import Pymem
import win32con
from typing import List

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

# Documentation: https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-VIRTUAL_ALLOC
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
    """
    This helper class handles the memory manipulation of the game.
    """

    def __init__(self, process_name="DarkSoulsIII.exe"):
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
        except:
            self.init = False

    def lazy_initalize(self):
        self.pid = _MemoryManipulator.get_pid(self.process_name)
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

    def require_initalize(func):

        def wrapper(self, *args, **kwargs):
            if not self.init:
                self.lazy_initalize()
                self.init = True
                self._inject_targeted_entity_tracking()

            return func(self, *args, **kwargs)

        return wrapper

    def clear_cache(self) -> None:
        """
        Clears the reference look-up cache of the memory manipulator.
        """
        self.cache = {}

    @staticmethod
    def get_pid(process_name: str) -> int:
        """
        Fetches the process PID of a process identified by a given name.

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
        """
        Resolves an address by its offsets and a base. Looks up the cache first.

        Args:
            addr_offsets: The offsets which will be resolved iteratively. The first offset is the
            offset to the base itself.
            base: The base offset from the start of the program's memory.

        Returns:
            The resolved address.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        # Look up the cache
        u_id = str((addr_offsets, base))
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
        """
        Reads an integer from memory and returns it.

        Args:
            address: The address to be looked into.

        Returns:
            The integer value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_long(address)

    @require_initalize
    def read_float(self, address) -> float:
        """
        Reads an integer from memory and returns it.

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
        """
        Reads a string from memory and returns it.

        Args:
            address: The address to be looked into.
            length: The expected (maximum) string length.
            null_term: Whether the string shall be cut after double 0x00.
            codec: The codec used to decode the bytes.

        Returns:
            The read string.

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
        """
        Reads raw bytes from memory and return them.

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
    def write_bit(self, address: int, index: int, value: int) -> None:
        """
        Writes a single bit.

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
    def write_int(self, address: int, value: int) -> None:
        """
        Writes an integer to memory.

        Args:
            address: The address to be written into.
            value: The value of the integer.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_long(self.pymem.process_handle, address, value)

    @require_initalize
    def write_float(self, address: int, value: float) -> None:
        """
        Writes a float to memory.

        Args:
            address: The address to be written into.
            value: The value of the float.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_float(self.pymem.process_handle, address, value)

    @require_initalize
    def write_bytes(self, address: int, buffer: bytes) -> None:
        """
        Writes a series of bytes to memory.

        Args:
            address: The first address to be written into.
            buffer: The bytes to write.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_bytes(self.pymem.process_handle, address, buffer, len(buffer))

    @require_initalize
    def activate_targeted_entity_info(self) -> None:
        """
        Injects the target entity info code pointer into the jump command at targeted entity info.
        """
        self.write_bytes(self.base_address + TARGETED_ENTITY_CODE_POS,
                         self.targeted_entity_injection)

    @require_initalize
    def deactivate_targeted_entity_info(self) -> None:
        """
        Injects the default game code pointer into the jump command at targeted entity info.
        """
        # reset volatile ptr
        self.write_bytes(self.target_ptr_volatile, bytes(8))
        self.write_bytes(self.base_address + TARGETED_ENTITY_CODE_POS, TARGETED_ENTITY_OLD_CODE)

    @require_initalize
    def _inject_targeted_entity_tracking(self) -> None:
        """
        Injects the targeted entity tracking binary into the running game.
        """
        # Make space for injection.
        # We need to fix this later, like wtf, currently writing into process memory, seems to work.
        inj_point = VIRTUAL_ALLOC(self.base_address + 0x10000, 80,
                                  win32con.MEM_RESERVE | win32con.MEM_COMMIT,
                                  win32con.PAGE_READWRITE)

        # Jump has 1 + 4 offset bytes.
        adr_offset = inj_point - (self.base_address + TARGETED_ENTITY_CODE_POS + 5)

        buff = b'\x48\x8b\x48\x58\x48\x8b\x89\x20\x03\x00\x00\x48\x81\xc1\x20\x74\x00\x00\x51\x48\xb9'
        buff += self.target_event.to_bytes(8, byteorder="little", signed=False)
        buff += b'\x48\x8f\x01\x48\xa3'
        buff += self.target_ptr.to_bytes(8, byteorder="little", signed=False)
        buff += b'\x48\xa3'
        buff += self.target_ptr_volatile.to_bytes(8, byteorder="little", signed=False)
        buff += b'\x48\x8b\x80\x90\x1f\x00\x00\xe9'

        # Offset Calculation: target: base_address + code + 7 (jmp and nop are one byte each)
        # Current: allocated_address + all_bytes_written + 4 (jump offset)
        jmp_adr = (self.base_address + TARGETED_ENTITY_CODE_POS + 7) - (inj_point + len(buff) + 4)

        buff += jmp_adr.to_bytes(4, byteorder="little", signed=True)  # Add jump address.

        self.targeted_entity_injection = b'\xe9' + adr_offset.to_bytes(
            4, byteorder="little", signed=True) + b'\x90\x90'
        pym.memory.write_bytes(self.pymem.process_handle, inj_point, buff, len(buff))


# MemoryManipulator is an already instanced class of type _MemoryManipulator!
# _MemoryManipulator instancing takes too long for single function calls, therefore we offer an
# already instanced object for functions to import

MemoryManipulator = _MemoryManipulator()
