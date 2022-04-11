"""Game control interface for key strokes and mouse clicks."""
import ctypes
from ctypes import wintypes
from typing import Any, List
import time

import numpy as np

from soulsgym.core.static import keybindings, keymap

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

    def __init__(self, *args: Any, **kwargs: Any):
        super(_KEYBDINPUT, self).__init__(*args, **kwargs)
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
    """Abstract in-game interaction by simulating keystrokes to the game."""

    def __init__(self):
        """Initialize the key state dictionary."""
        self.state = {key: False for key in keybindings.keys()}

    def update(self, actions: List):
        """Update the pressed keys state and execute key presses/releases.

        Args:
            actions: A list of pressed actions.
        """
        for action in self.state:
            if action in ("roll", "lightattack", "heavyattack", "parry") and action in actions:
                GameInput._press_key(keymap[keybindings[action]])
                time.sleep(0.02)
                GameInput._release_key(keymap[keybindings[action]])
                continue
            # nothing new, continue
            if self.state[action] == (action in actions):
                continue
            # key was not pressed before
            if not self.state[action]:
                self.state[action] = True
                GameInput._press_key(keymap[keybindings[action]])
            # key was pressed before
            elif self.state[action]:
                self.state[action] = False
                GameInput._release_key(keymap[keybindings[action]])

    def reset(self):
        """Reset the game input keys."""
        self.update([])

    def array_update(self, action_array: np.ndarray):
        """Interface update with boolean array encoded action selection.

        Args:
            action_array: The actions given as an boolean array. The order is 'forward', 'backward',
                'left', 'right', 'lightattack', 'roll', 'useitem', 'lockon'.
        """
        actions = [
            "forward", "backward", "left", "right", "lightattack", "roll", "useitem", "lockon"
        ]
        self.update([actions[i] for i in range(len(actions)) if action_array[i]])

    def restart(self):
        """Release all keys and sets the press state to False."""
        for action in self.state:
            if self.state[action]:
                self._release_key(keymap[keybindings[action]])
                self.state[action] = False

    def single_action(self, action: str, press_time: float = 0.15):
        """Perform a single action for a given amount of time.

        Args:
            action: The action that shall be performed (see the keybinding table)
            press_time: The time how long the key should be pressed.
        """
        GameInput._press_key(keymap[keybindings[action]])
        time.sleep(press_time)
        GameInput._release_key(keymap[keybindings[action]])

    @staticmethod
    def _press_key(key_hex_code: int):
        """Press a key identified by its hex code.

        Args:
            key_hex_code: The hex code to specify the key.
        """
        x = _INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=key_hex_code))
        USER32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    @staticmethod
    def _release_key(key_hex_code: int):
        """Release a key identified by its hex code.

        Args:
            key_hex_code: The hex code to specify the key.
        """
        x = _INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=key_hex_code, dwFlags=KEYEVENTF_KEYUP))
        USER32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
