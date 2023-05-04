![soulsgym_banner](https://raw.githubusercontent.com/amacati/SoulsGym/master/docs/img/soulsgym_banner.png)
[![PEP8 Check](https://github.com/amacati/SoulsGym/actions/workflows/github-actions.yaml/badge.svg)](https://github.com/amacati/SoulsGym/actions/workflows/github-actions.yaml)   [![Documentation Status](https://readthedocs.org/projects/soulsgym/badge/?version=latest)](https://soulsgym.readthedocs.io/en/latest/?badge=latest)

SoulsGym is an extension for OpenAI's [gym](https://github.com/Farama-Foundation/Gymnasium) toolkit for reinforcement learning environments. It enables training and testing of reinforcement learning algorithms on Dark Souls III bosses.
SoulsGym uses the game as a simulation that is modified at runtime by reading and writing into the game memory to create gym environments from the game's boss fights.

- [Requirements](#requirements)
- [Installation](#installation)
- [API](#api)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Requirements
You need to have the latest version of Dark Souls III installed since SoulsGym uses the game as engine for its environments. Dark Souls III is **not** available for free and has to be purchased (e.g. at the [Steam store](https://store.steampowered.com/app/374320/DARK_SOULS_III/)). SoulsGym requires the player to load into the game before any environments are created. In addition we require custom key bindings and graphic settings. It is also highly recommended to start the game in offline mode and delete your new save game after gym use to protect you from multiplayer bans.

> **Warning:** Please follow the setup description in our [official docs](https://soulsgym.readthedocs.io/en/latest/index.html) for the correct key settings, ban prevention, loss of game saves etc.

## Installation
To install soulsgym, use `pip install soulsgym`. At this time, we only support Windows. SoulsGym requires a running instance of Dark Souls III (see [requirements](#requirements)) and relies on the win32api. It is therefore not available on other operating systems.

## API
SoulsGym's environments follow the `gym` API as closely as possible. Since our environments are based on Dark Souls III we are, however, not able to provide reproducible results by setting the RNG seeds.

Our internal API for interacting with the game and creating the boss fight environments is located in the [`core`](soulsgym/core/) module. 

A detailed API documentation of our environments and the core library can be found in the [official docs](https://soulsgym.readthedocs.io/en/latest/index.html).

## Getting Started
You can use SoulsGym like any other `gym` environment. Below is an example of a random agent fighting against Iudex Gundyr:

```python
import gymnasium as gym
import soulsgym

env = gym.make("SoulsGymIudex-v0")
obs, info = env.reset()
terminated = False

while not terminated:
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

env.close()
```
> **Note:** Dark Souls III has to be running with the correct settings when executing the script. See [requirements](#requirements).
## Documentation
For details on the `soulsgym` package see our [official docs](https://soulsgym.readthedocs.io/en/latest/index.html).

## Contributing
If you'd like to contribute, feel free to reach out to me. In addition, have a look at the documentation and try to understand how the gym works.  

Implementing new bosses is probably the easiest way to contribute, and should be fairly self-contained. If you'd like to include other Souls games, definitely reach out first so that we can structure the project properly.
