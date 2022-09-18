import logging
from typing import NewType

import psutil
import pytest

from soulsgym.core.game_interface import Game

logger = logging.getLogger(__name__)

PositiveNumber = NewType("PositiveNumber", int)

attr_list = [("player_hp", 454), ("player_sp", 95), ("player_max_hp", 454), ("player_max_sp", 95),
             ("player_pose", None), ("player_animation", "Idle"), ("allow_player_death", True),
             ("player_stats", None), ("iudex_flags", True), ("iudex_hp", 1037),
             ("iudex_pose", None), ("iudex_animation", None), ("iudex_attacks", True),
             ("camera_pose", None), ("last_bonfire", None), ("allow_attacks", True),
             ("allow_hits", True), ("allow_moves", True), ("allow_deaths", True),
             ("allow_weapon_durability_dmg", True), ("lock_on", False), ("lock_on_bonus_range", 0),
             ("los_lock_on_deactivate_time", 2), ("time", PositiveNumber), ("global_speed", 1.0),
             ("gravity", True)]


@pytest.mark.parametrize("attr", attr_list)
def test_game_interface_attributes(attr):
    assert game_is_open()
    game = Game()
    print(f"Checking attribute {attr[0]}")
    output = getattr(game, attr[0])
    if attr[1] is None:
        ...
    elif attr[1] is PositiveNumber:
        assert output > 0, f"Output has to be a positive number. (Expected > 0, got {output})"
    else:
        assert output == attr[1], (f"Output of attribute {attr[0]} does not match expected value. "
                                   f"(Expected {attr[1]}, got {output})")
    print(f"Attribute {attr[0]}: {output}")


def game_is_open():
    for proc in psutil.process_iter():
        if proc.name() == "DarkSoulsIII.exe":
            return True
    return False
