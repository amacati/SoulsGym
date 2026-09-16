"""The ``game_input`` module provides an interface to trigger keystrokes from within the gym."""

import ctypes
import time
from typing import Any

from soulsgym.core.platform import SYSTEM, System
from soulsgym.core.static import keybindings, keymap

if SYSTEM is System.WINDOWS:
    from ctypes import wintypes

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    MAPVK_VK_TO_VSC = 0
    USER32 = ctypes.WinDLL("user32", use_last_error=True)
    wintypes.ULONG_PTR = wintypes.WPARAM

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", wintypes.ULONG_PTR),
        )

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", wintypes.ULONG_PTR),
        )

        def __init__(self, *args: Any, **kwargs: Any):
            super(_KEYBDINPUT, self).__init__(*args, **kwargs)
            if not self.dwFlags & KEYEVENTF_UNICODE:
                self.wScan = USER32.MapVirtualKeyExW(self.wVk, MAPVK_VK_TO_VSC, 0)

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = (
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        )

    class _INPUT(ctypes.Structure):
        class __INPUT(ctypes.Union):
            _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT))

        _anonymous_ = ("_input",)
        _fields_ = (("type", wintypes.DWORD), ("_input", __INPUT))


class _Win32Keyboard:
    """Windows keystroke backend using ``user32.SendInput`` with virtual-key codes."""

    def __init__(self, keymap: dict[str, int]):
        self.keymap = keymap  # char to Windows virtual-key code

    def press(self, key: str):
        vk = self.keymap[key]
        x = _INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=vk))
        USER32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    def release(self, key: str):
        vk = self.keymap[key]
        x = _INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=vk, dwFlags=KEYEVENTF_KEYUP))
        USER32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


class _X11Keyboard:
    """Linux keystroke backend using the X11 ``XTEST`` extension."""

    def __init__(self):
        self._x11 = ctypes.CDLL("libX11.so.6")
        self._xtst = ctypes.CDLL("libXtst.so.6")
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XKeysymToKeycode.restype = ctypes.c_ubyte
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XFlush.argtypes = [ctypes.c_void_p]
        self._xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError("Could not open X display")
        self._keycodes: dict[str, int] = {}  # Keycode cache

    def _keycode(self, key: str) -> int:
        kc = self._keycodes.get(key)
        if kc is None:
            kc = self._x11.XKeysymToKeycode(self._display, ord(key))
            self._keycodes[key] = kc
        return kc

    def press(self, key: str):
        self._xtst.XTestFakeKeyEvent(self._display, self._keycode(key), 1, 0)
        self._x11.XFlush(self._display)

    def release(self, key: str):
        self._xtst.XTestFakeKeyEvent(self._display, self._keycode(key), 0, 0)
        self._x11.XFlush(self._display)


class GameInput:
    """Trigger keystrokes for the game."""

    press_and_release_actions = ("roll", "lightattack", "heavyattack", "parry")
    DEFAULT_PRESS_TIME = 0.05

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
        if SYSTEM is System.WINDOWS:
            self._backend = _Win32Keyboard(self.keymap)
        elif SYSTEM is System.LINUX:
            self._backend = _X11Keyboard()
        else:
            raise NotImplementedError(f"GameInput not implemented for {SYSTEM}!")

    def add_action(self, action: str):
        """Queue a single action for the next game input.

        Action string has to be contained in :data:`.static.keybindings`.

        Args:
            action: The pressed action.
        """
        assert action in self.state.keys()
        self.queued_actions.append(action)

    def add_actions(self, actions: list[str]):
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
                self._press_key(self.keybindings[action])
            # key was pressed before
            elif self.state[action]:
                self.state[action] = False
                self._release_key(self.keybindings[action])
        # Process roll / hit / parry actions with blocking sleep
        for action in self.press_and_release_actions:
            if action in self.queued_actions:
                self.single_action(action, press_time=self.press_duration)
        self.queued_actions.clear()

    def reset(self):
        """Release all keys and set the press state to False."""
        for action in self.state:
            if self.state[action]:
                self._release_key(self.keybindings[action])
                self.state[action] = False
        self.queued_actions.clear()

    def single_action(self, action: str, press_time: float = 0.15):
        """Perform a single action for a given amount of time.

        Args:
            action: The action to trigger (see :data:`.static.keybindings`).
            press_time: The duration of the key press.
        """
        self._press_key(self.keybindings[action])
        time.sleep(press_time)
        self._release_key(self.keybindings[action])

    def _press_key(self, key: str):
        """Press a key identified by its binding character."""
        self._backend.press(key)

    def _release_key(self, key: str):
        """Release a key identified by its binding character."""
        self._backend.release(key)
