"""Tests for the Elden Ring game interface.

The live tests read the properties of the game interface from a running game. They are skipped if
the game is not running or the player is not loaded into the game. The skip condition is evaluated
lazily when a test runs, so collecting or deselecting the tests never attaches to a running game.
The live tests are classified by their effect on the game:

- ``integration``: Read-only tests. They use a game interface that only reads the game memory.
- ``integration`` and ``manual_destructive``: Tests that need the complete game interface. Its
  constructor injects the speedhack into the game and writes the game speed.

Run the live tests with Elden Ring in offline mode without EAC, the player loaded into the game and
standing in the arena of Margit, because the Margit properties read the entity of Margit.

The remaining tests run without a game. They check that the attribute lists cover all properties of
the game interface, that the expectations of the read-only tests hold for a simulated game and that
the setters and the game factory pass the correct arguments on.
"""

import inspect
import struct
from unittest.mock import Mock, call

import numpy as np
import pytest

from soulsgym.core.game_memory import GameMemory
from soulsgym.core.game_window import GameWindow
from soulsgym.exception import GameStateError, MemoryReadError
from soulsgym.games import game_factory
from soulsgym.games.eldenring import EldenRing
from tests.eldenring.utils import (
    attach_read_only,
    game_not_open,  # noqa: F401, used by the skipif conditions
    simulated_game,
)
from tests.utils import geq_assert, greater_than_assert, len_assert, shape_assert, type_assert

# The properties that read the game memory
memory_attributes = {
    "player_hp": {"type": int, ">": 0},
    "player_max_hp": {"type": int, ">": 0},
    "player_sp": {"type": int, ">=": 0},
    "player_max_sp": {"type": int, ">": 0},
    "player_mp": {"type": int, ">=": 0},
    "player_max_mp": {"type": int, ">": 0},
    "player_pose": {"type": np.ndarray, "shape": (4,)},
    "player_animation": {"type": int},
    "allow_player_death": {"type": bool},
    "player_stats": {"type": tuple, "len": 9},
    "margit_hp": {"type": int, ">=": 0},
    "margit_max_hp": {"type": int, ">": 0},
    "margit_pose": {"type": np.ndarray, "shape": (4,)},
    "margit_animation": {"type": str},
    "margit_animation_time": {"type": float},
    "margit_animation_max_time": {"type": float},
    "camera_pose": {"type": np.ndarray, "shape": (6,)},
    "is_ingame": {"type": bool},
    "last_bonfire": {"type": str},
    "allow_weapon_durability_dmg": {"type": bool},
    "lock_on": {"type": bool},
    "time": {"type": int, ">=": 0},
    "gravity": {"type": bool},
}
# The properties that need the speedhack or the game window of the complete game interface
interface_attributes = {
    "game_speed": {"type": float, ">=": 0},
    "img": {"type": np.ndarray, "shape": (90, 160, 3)},
    "img_resolution": {"type": tuple, "len": 2},
}


@pytest.fixture(scope="session")
def read_only_game():
    yield attach_read_only()


@pytest.fixture(scope="session")
def game():
    try:
        yield EldenRing()
    except (GameStateError, MemoryReadError, RuntimeError) as exc:
        pytest.skip(f"Elden Ring is not in a stable in-game state: {exc}")


def read_attribute(game: EldenRing, attr_name: str):
    """Read the attribute. Skip the test if the memory is not readable, e.g. after a reload."""
    try:
        return getattr(game, attr_name)
    except MemoryReadError as exc:
        pytest.skip(f"Elden Ring memory is no longer readable: {exc}")


def check_attribute(attr_name: str, attr, attr_info: dict):
    for attr_info_key, attr_info_value in attr_info.items():
        match attr_info_key:
            case "type":
                type_assert(attr_name, attr, attr_info_value)
            case ">":
                greater_than_assert(attr_name, attr, attr_info_value)
            case ">=":
                geq_assert(attr_name, attr, attr_info_value)
            case "shape":
                shape_assert(attr_name, attr, attr_info_value)
            case "len":
                len_assert(attr_name, attr, attr_info_value)
            case _:
                raise ValueError(f"Unknown attribute check category: {attr_info_key}")


@pytest.mark.integration
# Evaluated lazily so that collecting or deselecting the tests never attaches to a running game
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
@pytest.mark.parametrize(
    "attr_name, attr_info", memory_attributes.items(), ids=list(memory_attributes)
)
def test_read_only_game_attributes(read_only_game, attr_name, attr_info):
    check_attribute(attr_name, read_attribute(read_only_game, attr_name), attr_info)


@pytest.mark.integration
# The game fixture injects the speedhack and writes the game speed
@pytest.mark.manual_destructive
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
@pytest.mark.parametrize(
    "attr_name, attr_info", interface_attributes.items(), ids=list(interface_attributes)
)
def test_game_interface_attributes(game, attr_name, attr_info):
    check_attribute(attr_name, read_attribute(game, attr_name), attr_info)


def test_attribute_checks_cover_all_game_properties():
    properties = {
        name for name, member in inspect.getmembers(EldenRing) if isinstance(member, property)
    }

    assert not set(memory_attributes) & set(interface_attributes)
    assert properties == set(memory_attributes) | set(interface_attributes)


@pytest.mark.parametrize(
    "attr_name, attr_info", memory_attributes.items(), ids=list(memory_attributes)
)
def test_read_only_game_attributes_hold_for_a_simulated_game(attr_name, attr_info):
    check_attribute(attr_name, getattr(simulated_game(), attr_name), attr_info)


def test_read_only_attachment_only_builds_the_memory_reader(monkeypatch):
    memory = simulated_game().mem
    monkeypatch.setattr("tests.eldenring.utils.GameMemory", Mock(return_value=memory))

    game = attach_read_only()

    assert game.mem is memory
    assert game.is_ingame


SETTERS = {
    "player_hp": 10,
    "player_sp": 10,
    "player_mp": 10,
    "player_pose": [1.0, 2.0, 3.0, 0.5],
    "allow_player_death": False,
    "player_stats": [1] * 9,
    "margit_hp": 10,
    "margit_pose": [1.0, 2.0, 3.0, 0.5],
    "last_bonfire": "Stormhill - Castleward Tunnel",
    "gravity": False,
    "time": 5,
}


@pytest.mark.parametrize("name, value", SETTERS.items(), ids=list(SETTERS))
def test_setters_only_call_the_game_memory_with_arguments_it_accepts(name, value):
    game = simulated_game()

    setattr(game, name, value)

    assert any(call[0].startswith("write_") for call in game.mem.method_calls)


def test_player_pose_setter_rotates_the_player_to_the_target_angle():
    # Elden Ring interactions like entering fog gates require the player to face the object
    game = simulated_game()

    game.player_pose = [1.0, 2.0, 3.0, 1.48]

    record = game.data.addresses["PlayerLocalQ"]
    (rotation,) = [c.args[1] for c in game.mem.write_record.call_args_list if c.args[0] is record]
    for x, y, z, w in np.reshape(struct.unpack("8f", rotation), (2, 4)):
        assert (x, z) == (0, 0)
        assert 2 * np.arctan2(y, w) == pytest.approx(1.48)


class _ProcessMemory:
    """Byte memory that stands in for the game process backend of the ``GameMemory``."""

    def __init__(self, data: bytes):
        self.data = bytearray(data)

    def read_bytes(self, address: int, length: int) -> bytes:
        return bytes(self.data[address : address + length])

    def write_bytes(self, address: int, data: bytes):
        self.data[address : address + len(data)] = data


@pytest.mark.parametrize(
    "name, record", [("gravity", "PlayerGravity"), ("allow_player_death", "AllowPlayerDeath")]
)
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("initial_byte", [0b1010_0010, 0b1010_0011])
def test_flag_setters_write_the_disable_bit_of_the_resolved_address(
    name, record, flag, initial_byte
):
    """The flags are stored inverted in bit 0. The other bits of the byte are left untouched."""
    process = _ProcessMemory(bytes([initial_byte, 0, 0, 0]))
    memory = object.__new__(GameMemory)  # Skips attaching to the game process
    memory._backend = process
    memory.resolve_record = Mock(return_value=0)
    game = simulated_game()
    game.mem = memory

    setattr(game, name, flag)

    assert process.data[0] & 1 == (0 if flag else 1)
    assert process.data[0] & 0b1111_1110 == initial_byte & 0b1111_1110
    assert getattr(game, name) is flag
    # The setter and the getter both resolve the record of the flag
    assert memory.resolve_record.call_args_list == [call(game.data.addresses[record])] * 2


def test_img_resolution_is_the_image_resolution_of_the_game_window():
    window = object.__new__(GameWindow)  # Skips the screen capture setup
    window.game_id, window.game_resolution = "EldenRing", GameWindow.game_resolution
    window.img_height, window.img_width = 90, 160
    game = object.__new__(EldenRing)
    game._game_window = window

    assert game.img_resolution == (90, 160)

    game.img_resolution = (45, 80)

    assert game.img_resolution == (45, 80)


def test_reload_waits_until_the_respawned_player_is_idle(monkeypatch):
    # The game loads and the player rises at the grace before the player reacts to inputs again
    animations = iter([-1, 10004050, EldenRing.PLAYER_IDLE_ANIMATION])
    states = {
        "is_ingame": property(lambda game: True),
        "player_hp": property(lambda game: 455, lambda game, hp: None),
        "player_animation": property(lambda game: next(animations)),
        "allow_player_death": property(lambda game: True, lambda game, flag: None),
    }
    for name, state in states.items():
        monkeypatch.setattr(EldenRing, name, state)
    monkeypatch.setattr("soulsgym.games.eldenring.time.sleep", Mock())
    game = object.__new__(EldenRing)
    game.mem, game._game_flags, game._game_speed = Mock(), {}, 1.0

    game.reload()

    assert next(animations, None) is None


def test_game_factory_forwards_the_resolution_override(monkeypatch):
    elden_ring = Mock()
    monkeypatch.setattr("soulsgym.games.EldenRing", elden_ring)

    game = game_factory("EldenRing", game_resolution=(800, 450))

    assert game is elden_ring.return_value
    elden_ring.assert_called_once_with(game_resolution=(800, 450))
