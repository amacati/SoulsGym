![soulsgym_banner](https://raw.githubusercontent.com/amacati/SoulsGym/master/docs/img/soulsgym_banner.png)
[![PEP8 Check](https://github.com/amacati/SoulsGym/actions/workflows/github-actions.yaml/badge.svg)](https://github.com/amacati/SoulsGym/actions/workflows/github-actions.yaml)   [![Documentation Status](https://readthedocs.org/projects/soulsgym/badge/?version=latest)](https://soulsgym.readthedocs.io/en/latest/?badge=latest)

SoulsGym is an extension for [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), the successor of OpenAI's `gym` toolkit for reinforcement learning environments. It enables training and testing of reinforcement learning algorithms on boss fights from Dark Souls III, Elden Ring and other Souls games.
SoulsGym uses the games as simulations that are modified at runtime by reading and writing into the game memory to create gym environments from the games' boss fights.

- [Requirements](#requirements)
- [Installation](#installation)
- [API](#api)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Requirements
You need to have the latest version of Dark Souls III installed since SoulsGym uses the game as engine for its environments. Dark Souls III, Elden Ring etc. are **not** available for free and have to be purchased (e.g. at the [Steam store](https://store.steampowered.com/)). SoulsGym requires the player to load into the game before any environments are created. In addition we require custom key bindings and graphic settings. It is also highly recommended to start the game in offline mode and delete your new save game after gym use to protect you from multiplayer bans.

> **Warning:** Please follow the setup description in our [official docs](https://soulsgym.readthedocs.io/en/latest/index.html) for the correct key settings, ban prevention, loss of game saves etc.

## Installation
To install soulsgym, use `pip install soulsgym`. At this time, we only support Windows. SoulsGym requires a running instance of the game (see [requirements](#requirements)) and relies on the win32api. It is therefore not available on other operating systems.

## API
SoulsGym's environments follow the `gymnasium` API as closely as possible. Since our environments are based on the Souls games and Dark Souls III, Elden Ring etc. are nondeterministic, we are, however, not able to provide reproducible results by setting the RNG seeds.

Our internal API for interacting with the game and creating the boss fight environments is located in the [`core`](soulsgym/core/) module. 

A detailed API documentation of our environments and the core library can be found in the [official docs](https://soulsgym.readthedocs.io/en/latest/index.html).

## Getting Started
You can use SoulsGym like any other `gymnasium` environment. Below is an example of a random agent fighting against Iudex Gundyr:

```python
import gymnasium
import soulsgym

env = gymnasium.make("SoulsGymIudex-v0")
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

## Roadmap
This roadmap describes the planned future expansions and possible avenues for contributions. The project is designed to offer an extensive collection of boss fights across Souls games as gymnasium environments. However, as of this moment, only Iudex Gundyr from Dark Souls III is implemented and has been solved with a 45% winrate.

### Transitioning to image observations
Using ground truth information such as the exact player position, animation timings etc. drastically reduces the size of the observation space, which in general facilitates fast learning. For this reason, the first `soulsgym` environment, Iudex Gundyr from Dark Souls III, implements such an observation space. It verifies that it is indeed possible to train RL agents directly within the Souls games and learn a policy with meaningful winrates.

Nevertheless, human players do not have access to this information, but learn to play the game directly from images. In addition, agents that learn to act from rich visual input are more interesting from a research point of view, as they can be trained on much more general tasks. Furthermore, designing the obervation space with ground truth information can be tricky to implement as many boss fights in Dark Souls III, Elden Ring etc. include projectiles such as arrows or spells, ground effects or multiple entities that would need to be tracked. Images as a unified observation space would therefore drastically accelerate the development of new environments.

- [x] Add improved image capture module
- [x] Create Iudex Gundyr image observation space environment
- [ ] Train an agent with > 10% winrate

### Adding Elden Ring support
Initially, `soulsgym` was planned as a Dark Souls III RL extension only. However, much of the code such as game interfaces, player controls, window managers, speedhacks etc. are easily portable between the Souls games. In an effort to increase the number of available environments and the scope of the project, a major next step is the addition of an Elden Ring environment to the collection.

- [x] Refactor repository to support multiple Souls games
- [x] Add Elden Ring game interface
- [ ] Implement 'Margit, The Fell Omen' as gymnasium environment

### Increasing the number of environments
This one goes without saying. Having more bosses implemented is just better. There is not much focus on this though as long as we are transitioning to image observations, since the change of observation space also impacts how the environments have to be implemented.

## Contributing
If you'd like to contribute, feel free to reach out to me. In addition, have a look at the roadmap, the documentation and try to understand how the gym works.  

Implementing new bosses is probably the easiest way to contribute, and should be fairly self-contained. If you'd like to include other Souls games such as Dark Souls I or II, definitely reach out first so that we can coordinate our efforts.
