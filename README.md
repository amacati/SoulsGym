![soulsgym_banner](docs/img/soulsgym_banner.png)
[![PEP8 Check](https://github.com/amacati/SoulsGym/actions/workflows/github-actions.yaml/badge.svg)](https://github.com/amacati/SoulsGym/actions/workflows/github-actions.yaml)
SoulsGym is an extension for OpenAI's [gym](https://github.com/openai/gym) toolkit for reinforcement learning environments. It enables training and testing of reinforcement learning algorithms on boss fights from Dark Souls III.
SoulsGym uses Dark Souls III as a simulation that is modified at runtime by reading and writing into the game memory to create gym environments from the game's boss fights.

- [Requirements](#requirements)
- [Installation](#installation)
- [API](#api)
- [Getting Started](#getting-started)
- [Documentation](#documentation)

## Requirements
SoulsGym uses Dark Souls III as engine for its environments. You need to have the latest version of the game installed. Dark Souls III is **not** available for free and has to be purchased (e.g. at the [Steam store](https://store.steampowered.com/app/374320/DARK_SOULS_III/)). SoulsGym requires the player to load into the game before any environments are created. In addition we require custom key bindings and graphic settings. It is also highly recommended to start the game in offline mode and delete your new save game after gym use to protect you from multiplayer bans.

> **Warning:** Please follow the setup description in our [official docs](TODO:INSERT_DOC_LINK) for the correct key settings, ban prevention, loss of game saves etc.

## Installation
To install soulsgym, use `pip install soulsgym`. We support Python 3.9 and higher on Windows only. SoulsGym requires a running instance of Dark Souls III (see [requirements](#requirements)). It is therefore not available on other operating systems.

## API
SoulsGym's environments follow the `gym` API as closely as possible. Since our environments are based on Dark Souls III we are, however, not able to provide reproducible results by setting the RNG seeds.

Our internal API for interacting with the game and creating the boss fight environments is located in the [`core`](soulsgym/core/) module. 

A detailed API documentation of our environments and the core library can be found in the [official docs](TODO:INSERT_DOCS_LINK).

## Getting Started
You can use SoulsGym like any other `gym` environment. Below is an example of a random agent fighting against Iudex Gundyr:

```python
import gym
import soulsgym

env = gym.make("SoulsGymIudex-v0")
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

env.close()
```
> **Note:** Dark Souls III has to be running with the correct settings when executing the script. See [requirements](#requirements).
## Documentation
For details on the `soulsgym` package see our [official docs](TODO:INSERT_DOCS_LINK).