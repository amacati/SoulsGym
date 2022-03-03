"""Read various tables and make them available as python objects.

Todo:
    * Clean up this mess. Highly unstructured at the moment.
"""
from pathlib import Path
import yaml
from soulsgym.envs.utils.onehot import OneHotEncoder

config_path = Path(__file__).resolve().parent / "tables.yaml"

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

keybindings = config["keybindings"]

# msdn.microsoft.com/en-us/library/dd375731
keymap = config["keymap"]

# All unknown animations are binned into a zero vector from the OneHotEncoder
phase1_animations = [x for x in config["phase1_animations"]]
p1_anim_enc = OneHotEncoder(allow_unknown=True)
p1_anim_enc.fit(phase1_animations)

action_list = [x for x in config["action_list"]]
# Each action gets an array compatible to GameInput.array_update which is stored in actions
actions = []
for action in action_list:
    action_array = [False] * len(keybindings)
    for idx, action_key in enumerate(keybindings.values()):
        action_array[idx] = action_key in action
    actions.append(action_array)

coordinates = config["coordinates"]

player_animations = config["player_animations"]
