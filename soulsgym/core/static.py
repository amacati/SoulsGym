"""The ``soulsgym.core.static`` module is a collection of all module constants.

We read all config files into dictionaries and make them available as Python objects. The static
collection is useful beyond the ``core`` module itself. We provide a complete list of possible
values for boss and player animations. These can be employed to fit one-hot encoders to animation
names prior to learning.
"""
from pathlib import Path

import yaml
import numpy as np

root = Path(__file__).resolve().parent / "data"

with open(root / "keys.yaml", "r") as f:
    keys = yaml.load(f, Loader=yaml.SafeLoader)

#: Dictionary mapping of player actions to keyboard keys.
keybindings = keys["binding"]

#: Dictionary mapping of keyboard keys to Windows virtual-key codes. See `virtual-key codes docs
#: <https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes>`_.
keymap = keys["keymap"]  # msdn.microsoft.com/en-us/library/dd375731

with open(root / "actions.yaml", "r") as f:
    #: Dictionary mapping of integers to action combinations.
    actions = yaml.load(f, Loader=yaml.SafeLoader)

with open(root / "coordinates.yaml", "r") as f:
    #: Dictionary mapping of game coordinates for each boss fight.
    coordinates = yaml.load(f, Loader=yaml.SafeLoader)
# Numpify all coordinates
for boss in coordinates.keys():
    for key in coordinates[boss].keys():
        coordinates[boss][key] = np.array(coordinates[boss][key])

with open(root / "animations.yaml", "r") as f:
    _animations = yaml.load(f, Loader=yaml.SafeLoader)

#: Dictionary of player animations. Animations are separated into ``standard``, ``critical`` and
#: ``all``. ``critical`` animations should not occur during normal operation and can be disregarded
#: by users. All animations have an animation timing during which the player cannot take any action.
player_animations = _animations["player"]
player_animations["all"] = player_animations["standard"] | player_animations["critical"]

#: Dictionary of boss animations. Each boss has its own dictionary accessed by its boss ID.
#: Individual boss animations are separated into ``attacks``, ``movement`` and ``all``.
boss_animations = _animations["boss"]
for boss_anim in boss_animations.values():
    boss_anim["all"] = boss_anim["attacks"] + boss_anim["movement"]

with open(root / "player_stats.yaml", "r") as f:
    #: Dictionary of player stats for each boss fight. Player stats are mapped by boss ID.
    player_stats = yaml.load(f, Loader=yaml.SafeLoader)

with open(root / "bonfires.yaml", "r") as f:
    #: Dictionary mapping of bonfire IDs to ingame integer IDs.
    bonfires = yaml.load(f, Loader=yaml.SafeLoader)

with open(root / "addresses.yaml", "r") as f:
    _addresses = yaml.load(f, Loader=yaml.SafeLoader)
    #: Dictionary of recurring initial base address offset from the game's ``base_address``
    address_bases = _addresses["bases"]
    #: Dictionary of the list of address offsets for the pointer chain to each game property's
    #: memory location
    address_offsets = _addresses["address_offsets"]
    address_base_patterns = _addresses["bases_by_pattern"]
