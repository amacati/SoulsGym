from functools import cache
import warnings

import pytest
import numpy as np

from soulsgym.core.utils import get_pid
from soulsgym.games.eldenring import EldenRing

from tests.utils import type_assert, greater_than_assert, geq_assert, shape_assert, len_assert


@cache
def game_not_open():
    try:
        get_pid("eldenring.exe")
        e = EldenRing()
        if not e.is_ingame:
            warnings.warn("EldenRing: Player is not in-game.")
            return True
        return False
    except RuntimeError:
        return True


@pytest.fixture(scope="session")
def game():
    yield EldenRing()


game_attributes = {"player_hp": {"type": int, ">": 0},
                   "player_max_hp": {"type": int, ">": 0},
                   "player_sp": {"type": int, ">=": 0},
                   "player_max_sp": {"type": int, ">": 0},
                   "player_mp": {"type": int, ">=": 0},
                   "player_max_mp": {"type": int, ">": 0},
                   "player_pose": {"type": np.ndarray, "shape": (4, )},
                   "player_animation": {"type": int},
                   "allow_player_death": {"type": bool},
                   "player_stats": {"type": tuple, "len": 9},
                   # "iudex_flags": {"type": bool},
                   # "iudex_hp": {"type": int, ">": 0},
                   # "iudex_pose": {"type": np.ndarray, "shape": (4, )},
                   # "iudex_animation": {"type": str},
                   # "iudex_attacks": {"type": bool},
                   # "camera_pose": {"type": np.ndarray, "shape": (6, )},
                   "last_bonfire": {"type": str},
                   # "allow_attacks": {"type": bool},
                   # "allow_hits": {"type": bool},
                   # "allow_moves": {"type": bool},
                   # "allow_deaths": {"type": bool},
                   # "allow_weapon_durability_dmg": {"type": bool},
                   "lock_on": {"type": bool},
                   # "lock_on_bonus_range": {"type": float, ">=": 0},
                   # "los_lock_on_deactivate_time": {"type": float, ">=": 0},
                   "time": {"type": int, ">=": 0},
                   "game_speed": {"type": float, ">=": 0},
                   "gravity": {"type": bool},
                   }


@pytest.mark.skipif(game_not_open(), reason="Dark Souls III is not running.")
@pytest.mark.parametrize("attr_name, attr_info", game_attributes.items())
def test_game_attributes(game, attr_name, attr_info):
    attr = getattr(game, attr_name)
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
