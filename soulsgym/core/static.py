"""The ``soulsgym.core.static`` module is a collection of all module constants.

We read all config files into dictionaries and make them available as Python objects. The static
collection is useful beyond the ``core`` module itself. We provide a complete list of possible
values for boss and player animations. These can be employed to fit one-hot encoders to animation
names prior to learning.
"""

from pathlib import Path

import numpy as np
import yaml

_games = {"DarkSoulsIII": "darksouls3", "EldenRing": "eldenring"}  # Game ID and location

_data_paths = {
    game: Path(__file__).resolve().parent / "data" / game_location
    for game, game_location in _games.items()
}


def _load_keybindings_and_mapping() -> tuple[dict, dict]:
    keybindings, keymap = {}, {}
    for game in _games:
        with open(_data_paths[game] / "keys.yaml", "r") as f:
            keys = yaml.load(f, Loader=yaml.SafeLoader)

        keybindings[game] = keys["binding"]
        keymap[game] = keys["keymap"]  # msdn.microsoft.com/en-us/library/dd375731
    return keybindings, keymap


def _load_actions() -> dict:
    actions = {}
    for game in _games:
        with open(_data_paths[game] / "actions.yaml", "r") as f:
            actions[game] = yaml.load(f, Loader=yaml.SafeLoader)
    return actions


def _load_coordinates() -> dict:
    coordinates = {}
    for game in _games:
        with open(_data_paths[game] / "coordinates.yaml", "r") as f:
            coords = yaml.load(f, Loader=yaml.SafeLoader)

        for boss in coords.keys():  # Numpify all coordinates
            for key in coords[boss].keys():
                coords[boss][key] = np.array(coords[boss][key])

        coordinates[game] = coords
    return coordinates


def _load_animations() -> tuple[dict, dict, dict]:
    player_animations, critical_player_animations, boss_animations = {}, {}, {}
    for game in _games:
        with open(_data_paths[game] / "animations.yaml", "r") as f:
            animations = yaml.load(f, Loader=yaml.SafeLoader)

        _player_animations = animations["player"]["standard"]
        for i, animation in enumerate(_player_animations):
            _player_animations[animation] = {"timings": _player_animations[animation], "ID": i}

        player_animations[game] = _player_animations
        critical_player_animations[game] = animations["player"]["critical"]

        _boss_animations = animations["boss"]
        for _boss_animation in _boss_animations.values():
            _boss_animation["all"] = {}
            i = 0
            for animation in _boss_animation["attacks"]:
                _boss_animation["all"][animation] = {"ID": i, "type": "attacks"}
                i += 1
            for animation in _boss_animation["movement"]:
                _boss_animation["all"][animation] = {"ID": i, "type": "movement"}
                i += 1
            for animation in _boss_animation["misc"]:
                _boss_animation["all"][animation] = {"ID": i, "type": "misc"}
                i += 1

        boss_animations[game] = _boss_animations
    return player_animations, critical_player_animations, boss_animations


def _load_player_stats() -> dict:
    player_stats = {}
    for game in _games:
        with open(_data_paths[game] / "player_stats.yaml", "r") as f:
            player_stats[game] = yaml.load(f, Loader=yaml.SafeLoader)
    return player_stats


def _load_bonfires() -> dict:
    bonfires = {}
    for game in _games:
        with open(_data_paths[game] / "bonfires.yaml", "r") as f:
            bonfires[game] = yaml.load(f, Loader=yaml.SafeLoader)
    return bonfires


def _load_addresses() -> tuple[dict, dict, dict]:
    address_bases, addresses, address_base_patterns = {}, {}, {}
    for game in _games:
        with open(_data_paths[game] / "addresses.yaml", "r") as f:
            adresses = yaml.load(f, Loader=yaml.SafeLoader)

        address_bases[game] = adresses["bases"]
        addresses[game] = adresses["addresses"]
        address_base_patterns[game] = adresses["bases_by_pattern"]
    return address_bases, addresses, address_base_patterns


# Initialize all static data dictionaries. Each dictionary is indexed by the game's name.
#: Dictionary mapping of player actions to keyboard keys.
keybindings = {}
#: Dictionary mapping of keyboard keys to Windows virtual-key codes. See `virtual-key codes docs
#: <https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes>`_.
keymap = {}

keybindings, keymap = _load_keybindings_and_mapping()  # Load here to allow documentation

#: Dictionary mapping of integers to action combinations.
actions = _load_actions()

#: Dictionary mapping of game coordinates for each boss fight.
coordinates = _load_coordinates()

#: Dictionary of player animations. All animations have an animation timing during which the player
#: cannot take any action, and a unique ID.
player_animations = {}
#: ``critical`` animations should not occur during normal operation and can be disregarded by users.
#: They require special recovery handling by the gym.
critical_player_animations = {}
#: Dictionary of boss animations. Each boss has its own dictionary accessed by its boss ID.
#: Individual boss animations are separated into ``attacks``, ``movement`` and ``all``. ``all``
#: animations have a unique ID.
boss_animations = {}

player_animations, critical_player_animations, boss_animations = _load_animations()

#: Dictionary of player stats for each boss fight. Player stats are mapped by boss ID.
player_stats = _load_player_stats()

#: Dictionary mapping of bonfire IDs to ingame integer IDs.
bonfires = _load_bonfires()

#: Dictionary of recurring initial base address offset from the game's ``base_address``
address_bases = {}
#: Dictionary of address offsets for the pointer chain to each game property's memory location
addresses = {}
#: Dictionary of patterns that can be scanned by AOB modules to locate the base addresses
address_base_patterns = {}

address_bases, addresses, address_base_patterns = _load_addresses()
