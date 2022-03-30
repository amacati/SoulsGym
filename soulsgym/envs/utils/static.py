"""Read various static collections and make them available as python objects."""
from pathlib import Path
import yaml

root = Path(__file__).resolve().parent / "data"

with open(root / "keys.yaml", "r") as f:
    keys = yaml.load(f, Loader=yaml.SafeLoader)

# Keys
keybindings = keys["binding"]
keymap = keys["keymap"]  # msdn.microsoft.com/en-us/library/dd375731

# Actions
with open(root / "actions.yaml", "r") as f:
    actions = yaml.load(f, Loader=yaml.SafeLoader)

# Coordinates
with open(root / "coordinates.yaml", "r") as f:
    coordinates = yaml.load(f, Loader=yaml.SafeLoader)

# Animations
with open(root / "animations.yaml", "r") as f:
    animations = yaml.load(f, Loader=yaml.SafeLoader)

player_animations = animations["player"]
iudex_animations = animations["iudex"]
