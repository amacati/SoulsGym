import ctypes
from ctypes import wintypes
import time
import numpy as np

from soulsgym.envs.utils.tables import keybindings, keymap

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MAPVK_VK_TO_VSC = 0

USER32 = ctypes.WinDLL('user32', use_last_error=True)

wintypes.ULONG_PTR = wintypes.WPARAM


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo",
                                                                        wintypes.ULONG_PTR))


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(_KEYBDINPUT, self).__init__(*args, **kwds)
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = USER32.MapVirtualKeyExW(self.wVk, MAPVK_VK_TO_VSC, 0)


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD))


class _INPUT(ctypes.Structure):

    class __INPUT(ctypes.Union):
        _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT))

    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD), ("_input", __INPUT))


LPINPUT = ctypes.POINTER(_INPUT)


class GameInput:
    """
    A class that abstracts in-game interaction by simulating keystrokes to the game.
    """

    def __init__(self):
        self.state = {
            'Forward': False,
            'Backward': False,
            'Left': False,
            'Right': False,
            'LightAttack': False,
            'Roll': False,
            'UseItem': False,
            'LockOn': False
        }

    def update(self, new_state: dict) -> None:
        """
        Updates the pressed keys such that the current state represents the given state.

        Args:
            new_state: The state dict for the new input state.
        """
        for action in self.state:
            if action == "Roll" and new_state[action]:
                GameInput._press_key(keymap[keybindings[action]])
                time.sleep(0.02)
                GameInput._release_key(keymap[keybindings[action]])
                continue
            # nothing new, continue
            if self.state[action] == new_state[action]:
                continue
            # key was not pressed before
            if not self.state[action]:
                self.state[action] = True
                GameInput._press_key(keymap[keybindings[action]])
            # key was pressed before
            elif self.state[action]:
                self.state[action] = False
                GameInput._release_key(keymap[keybindings[action]])

    def array_update(self, array_state: np.ndarray) -> None:
        """
        Interfaces update with boolean array encoded action selection.

        Args:
            array_update: The states given as an boolean array. The order is 'Forward', 'Backward',
            'Left', 'Right', 'LightAttack', 'Roll', 'UseItem', 'LockOn'.
        """
        buff_dict = {
            'Forward': bool(array_state[0]),
            'Backward': bool(array_state[1]),
            'Left': bool(array_state[2]),
            'Right': bool(array_state[3]),
            'LightAttack': bool(array_state[4]),
            'Roll': bool(array_state[5]),
            'UseItem': bool(array_state[6]),
            'LockOn': bool(array_state[7])
        }
        self.update(buff_dict)

    def restart(self) -> None:
        """
        Releases all keys and sets the press state to False.
        """
        for action in self.state:
            if self.state[action]:
                self._release_key(keymap[keybindings[action]])
                self.state[action] = False

    def single_action(self, action: str, press_time: float = 0.15) -> None:
        """
        Performs a single action for a given amount of time.

        Args:
            action: The action that shall be performed (see the keybinding table)
            press_time: The time how long the key should be pressed.
        """
        GameInput._press_key(keymap[keybindings[action]])
        time.sleep(press_time)
        GameInput._release_key(keymap[keybindings[action]])

    @staticmethod
    def _press_key(key_hex_code: int) -> None:
        """
        Presses a key identified by its hex code.

        Args:
            key_hex_code: The hex code to specify the key.
        """
        x = _INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=key_hex_code))
        USER32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    @staticmethod
    def _release_key(key_hex_code: int) -> None:
        """
        Releases a key identified by its hex code.

        Args:
            key_hex_code: The hex code to specify the key.
        """
        x = _INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=key_hex_code, dwFlags=KEYEVENTF_KEYUP))
        USER32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
