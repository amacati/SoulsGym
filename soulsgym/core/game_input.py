"""The ``game_input`` module provides an interface to trigger keystrokes from within the gym."""
import ctypes
from ctypes import wintypes
from typing import Any, List
import time
import platform

from soulsgym.core.static import keybindings, keymap

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MAPVK_VK_TO_VSC = 0

if platform.system() == "Windows":  # Guard to prevent WinDLL to load on non-Windows systems
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


class GameInput:
    """Trigger keystrokes by calling the Windows user32 API."""

    press_and_release_actions = ("roll", "lightattack", "heavyattack", "parry")
    DEFAULT_PRESS_TIME = 0.015

    def __init__(self, game_id: str, game_speed: float = 1.0):
        """Initialize the key state dictionary.

        Args:
            game_id: The name of the game.
            game_speed: The speed of the game. We need to adapt the duration of a key press to the
                game speed, because the game recognizes a key press only if it is held for a certain
                amount of in-game time.
        """
        self.keybindings = keybindings[game_id]
        self.keymap = keymap[game_id]
        self.state = {key: False for key in self.keybindings.keys()}
        self.queued_actions = []
        self.press_duration = self.DEFAULT_PRESS_TIME / game_speed

    def add_action(self, action: str):
        """Queue a single action for the next game input.

        Action string has to be contained in :data:`.static.keybindings`.

        Args:
            actions: The pressed action.
        """
        assert action in self.state.keys()
        self.queued_actions.append(action)

    def add_actions(self, actions: List[str]):
        """Queue multiple actions for the next game input.

        Action strings have to be contained in :data:`.static.keybindings`.

        Args:
            actions: A list of pressed actions.
        """
        for action in actions:
            assert action in self.state.keys()
        self.queued_actions.extend(actions)

    def update_input(self):
        """Update the pressed keys state with queued actions and execute key presses/releases.

        Actions have to be queued with :meth:`.GameInput.add_action` and
        :meth:`.GameInput.add_actions`. Some actions (e.g. rolling) require an immediate release
        after pressing the key, or else the player would perform a different action such as running.
        All other keystrokes remain pressed as long as successive updates contain the corresponding
        action (e.g. running).
        """
        for action in self.state:
            # If roll and a direction are specified, the roll press has to come after the direction
            # key. We therefore handle press_and_release_actions at the end of the function
            if action in self.press_and_release_actions and action in self.queued_actions:
                continue
            # nothing new, continue
            if self.state[action] == (action in self.queued_actions):
                continue
            # key was not pressed before
            if not self.state[action]:
                self.state[action] = True
                self._press_key(self.keymap[self.keybindings[action]])
            # key was pressed before
            elif self.state[action]:
                self.state[action] = False
                self._release_key(self.keymap[self.keybindings[action]])
        # Process roll / hit / parry actions with blocking sleep
        for action in self.press_and_release_actions:
            if action in self.queued_actions:
                self.single_action(action, press_time=self.press_duration)
        self.queued_actions.clear()

    def reset(self):
        """Release all keys and set the press state to False."""
        for action in self.state:
            if self.state[action]:
                self._release_key(self.keymap[self.keybindings[action]])
                self.state[action] = False
        self.queued_actions.clear()

    def single_action(self, action: str, press_time: float = 0.15):
        """Perform a single action for a given amount of time.

        Args:
            action: The action to trigger (see :data:`.static.keybindings`).
            press_time: The duration of the key press.
        """
        self._press_key(self.keymap[self.keybindings[action]])
        time.sleep(press_time)
        self._release_key(self.keymap[self.keybindings[action]])

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
