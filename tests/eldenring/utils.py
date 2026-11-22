"""Helpers for the tests of Elden Ring: the attachment to a live game and fakes of the game."""

import warnings
from functools import cache
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np

from soulsgym.core.game_memory import GameMemory
from soulsgym.core.utils import get_pid
from soulsgym.exception import GameStateError, MemoryReadError
from soulsgym.games.eldenring import EldenRing
from soulsgym.games.game import StaticGameData


def attach_read_only() -> EldenRing:
    """Attach a game interface to the running game that does not change the game.

    The constructor of the game interface injects the speedhack into the game and writes the game
    speed. We build the interface from the static data and the memory reader instead. All
    properties that read the game memory work on it. The properties that need the speedhack or the
    game window do not.

    Returns:
        The game interface.
    """
    game = object.__new__(EldenRing)
    game.data = StaticGameData(EldenRing.game_id)
    game.mem = GameMemory(process_name=EldenRing.process_name)
    game.mem.clear_cache()  # The memory reader is a singleton that might hold old addresses
    return game


@cache
def game_not_open() -> bool:
    """Return True when Elden Ring is not running or the player is not loaded into the game.

    The check only reads from the game. The live tests evaluate it lazily through string ``skipif``
    conditions, so collecting or deselecting them never attaches to a running game.
    """
    try:
        get_pid(EldenRing.process_name)
        if not attach_read_only().is_ingame:
            warnings.warn("EldenRing: Player is not in-game.")
            return True
        return False
    except (GameStateError, MemoryReadError, RuntimeError):
        return True


def simulated_game(**values: int | float | bytes) -> EldenRing:
    """Create a game interface for a simulated game without a game process.

    The memory reader follows the ``GameMemory`` API. Calls that do not match it raise a
    ``TypeError``. All records hold small values unless ``values`` overrides them by record name.

    Args:
        values: The values of the records that should not hold a small default value.

    Returns:
        The game interface.
    """
    game = object.__new__(EldenRing)
    game.data = StaticGameData(EldenRing.game_id)
    names = {id(record): name for name, record in game.data.addresses.items()}
    values = {"LastGrace": next(iter(game.data.bonfires.values())), **values}

    def read_record(record: dict) -> int | float | bytes:
        name = names[id(record)]
        if name in values:
            return values[name]
        return {"int": 1, "float": 0.5, "bytes": bytes(record.get("length", 0))}[record["type"]]

    game.mem = create_autospec(GameMemory, instance=True)
    game.mem.read_record.side_effect = read_record
    game.mem.resolve_record.return_value = 0x1000
    game.mem.read_int.return_value = 1
    return game


COORDS = StaticGameData(EldenRing.game_id).coordinates["margit"]
PLAYER_MAX_HP = 455
BOSS_MAX_HP = 4174
MARGIT_IDLE_ANIMATION = 2000020
PLAYER_IDLE_ANIMATION = 10000000


class FakeGame:
    """Stand-in for the Elden Ring interface that logs attribute writes and method calls."""

    TICK_MS = 10  # Ingame time that passes with every read of the game time

    # The fake reuses the parts of the game interface that do not need a game process. The
    # translation of the animation reads the raw animation ID of Margit from the fake memory.
    timed = staticmethod(EldenRing.timed)
    PLAYER_IDLE_ANIMATION = EldenRing.PLAYER_IDLE_ANIMATION
    reset_player_hp = EldenRing.reset_player_hp
    reset_player_sp = EldenRing.reset_player_sp
    reset_player_mp = EldenRing.reset_player_mp
    reset_boss_hp = EldenRing.reset_boss_hp
    margit_animation = EldenRing.margit_animation
    _boss_animation_names = EldenRing._boss_animation_names
    # The camera pose is controlled by game input, so writes do not change the pose we read
    camera_pose = property(lambda self: np.zeros(6), lambda self, normal: None)

    def __init__(self):
        clock = SimpleNamespace(now=0.0)  # The clock only advances when the fake game sleeps
        clock.time = lambda: clock.now
        self.__dict__.update(
            data=StaticGameData(EldenRing.game_id),
            mem=Mock(read_record=Mock(return_value=MARGIT_IDLE_ANIMATION)),
            clock=clock,
            writes=[],
            calls=[],
            is_ingame=True,
            game_speed=1.0,
            lock_on=False,
            player_animation=PLAYER_IDLE_ANIMATION,
            margit_animation_time=0.0,
            player_max_hp=PLAYER_MAX_HP,
            player_hp=PLAYER_MAX_HP,
            player_max_sp=100,
            player_sp=100,
            player_max_mp=50,
            player_mp=50,
            player_pose=COORDS["bonfire"].copy(),
            margit_max_hp=BOSS_MAX_HP,
            margit_hp=BOSS_MAX_HP,
            margit_pose=np.zeros(4),
            _time=0,
        )

    def __setattr__(self, name: str, value: object):
        if not name.startswith("_"):
            self.writes.append(name)
        super().__setattr__(name, value)

    @property
    def time(self) -> int:
        """The ingame time in ms. The fake game loop advances by one tick with every read."""
        self._time += self.TICK_MS
        return self._time

    @time.setter
    def time(self, value: int):
        self._time = value

    def deal_damage(self, player: int = 0, boss: int = 0):
        """Lower the HP in the game like hits that land during the next step."""
        self.__dict__.update(player_hp=self.player_hp - player, margit_hp=self.margit_hp - boss)

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def reload(self):
        self.calls.append("reload")

    def clear_cache(self):
        self.calls.append("clear_cache")

    def sleep(self, t: float):
        self.clock.now += t


class FakeInput:
    """Fake game input that logs the actions and lets tests react to them."""

    def __init__(self):
        self.calls = []
        self.on_action = {}  # Callbacks that simulate the reaction of the game to a single action
        self.queued = []
        self.updates = []  # The actions that each input update has executed

    def reset(self):
        self.calls.append("reset")
        self.queued = []

    def add_action(self, action: str):
        self.queued.append(action)

    def add_actions(self, actions: list[str]):
        self.queued.extend(actions)

    def update_input(self):
        self.calls.append("update_input")
        self.updates.append(self.queued)
        self.queued = []

    def single_action(self, action: str, *args: float):
        self.calls.append(action)
        if action in self.on_action:
            self.on_action[action]()
