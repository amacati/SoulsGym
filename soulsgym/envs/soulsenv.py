"""The ``SoulsEnv`` class is the abstract base class for all ``soulsgym`` environments.

It includes the general gym logic and defines abstract methods that all environments have to
implement.
"""
import time
import logging
from typing import Tuple, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from argparse import Namespace

import gym
import yaml
import numpy as np
from pymem.exception import MemoryReadError

from soulsgym.core.game_input import GameInput
from soulsgym.core.logger import Logger, GameState
from soulsgym.core.game_interface import Game
from soulsgym.core.static import coordinates, actions, player_animations, player_stats
from soulsgym.core.static import boss_animations
from soulsgym.core.game_window import GameWindow
from soulsgym.exception import GameStateError, ResetNeeded, InvalidPlayerStateError

logger = logging.getLogger(__name__)


class SoulsEnv(gym.Env, ABC):
    """Abstract base class for ``soulsgym`` environments.

    Each ``SoulsEnv`` initializes a :class:`.GameInput`, a :class:`.GameWindow`, a :class:`.Game`
    interface and a :class:`.Logger` to read from and to the game. During an episode the game is
    paused per default. At each :meth:`.SoulsEnv.step` call the environment applies the current
    input if valid (see :meth:`.SoulsEnv._apply_action` for details). It then unpauses the game,
    waits for 0.1 seconds and pauses again. The new game state is logged and processed to update the
    internal game state. The environment tries to detect and recover from any unexpected deviations.

    To solve any camera control issues we lock on to the boss at all times. For exceptions see
    :meth:`.SoulsEnv._lock_on`.

    Note:
        We deem it too complicated to learn from the image alone. The :class:`.GameWindow` class is
        however designed to provide this capability and the gym can easily be extended to yield
        image data as well.

    Warning:
        Dark Souls III has to be running at the initialization of :class:`SoulsEnv`.
    """

    metadata = {'render.modes': ['human']}
    ENV_ID = ""  # Each SoulsGym has to define its own ID and name the config files accordingly
    _step_size = 0.1

    def __init__(self):
        """Initialize the game managers, load the environment config and set the game properties."""
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(actions))
        self._internal_state = None
        self.done = False
        self._game_input = GameInput()
        self._game_window = GameWindow()
        self._game_check()
        self.game = Game()
        try:
            self._game_logger = Logger(self.ENV_ID)
        except MemoryReadError:
            logger.error("__init__: Player is not loaded into the game")
            raise GameStateError("Player is not loaded into the game")
        self.config_path = Path(__file__).parent / "config"
        self.env_args = self._load_env_args()
        self._set_game_properties()
        self._is_init = False
        logger.info(self.env_args.init_msg)
        logger.debug("Env init complete")

    @abstractmethod
    def reset(self) -> GameState:
        """Reset the environment to the beginning of an episode.

        Returns:
            The first game state after a reset.
        """

    @abstractmethod
    def _env_setup(self):
        """Execute the setup sequence for the boss fight."""

    @abstractmethod
    def compute_reward(self, game_state: GameState) -> float:
        """Compute the reward from a game state.

        Args:
            game_state: A game state.

        Returns:
            The reward for the provided game state.
        """

    def step(self, action: int) -> Tuple[GameState, float, bool, dict]:
        """Perform a step forward in the environment with a given action.

        Each step advances the ingame time by 0.1s. The game is halted before and after the step.

        Args:
            action: The action that is applied during this step.

        Returns:
            A tuple of the next game state, the reward, the done flag and an additional info
            dictionary.

        Raises:
            ResetNeeded: `step()` was called after the episode was already finished.
        """
        if self.done:
            logger.error("step: Environment step called after environment was done")
            raise ResetNeeded("Environment step called after environment was done")
        self._apply_action(action)
        self._step()
        reward = self.compute_reward(self._internal_state)
        if self.done:
            logger.debug("step: Episode finished")
        return self._internal_state, reward, self.done, {}

    def close(self):
        """Unpause the game, reset altered game properties and reload.

        Note:
            Does not wait for reload to complete.
        """
        self.game.resume_game()
        self._game_input.reset()
        # Restore game parameter defaults
        self.game.lock_on_range = 15
        self.game.los_lock_on_deactivate_time = 2
        self.game.allow_moves = True
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_deaths = True
        self.game.allow_weapon_durability_dmg = True
        self.game.player_hp = 0  # Kill player to force game reload. Don't wait for completion
        logger.debug("SoulsEnv close successful")

    def _game_check(self):
        """Check if the game is currently running."""
        self._game_window._get_ds_app_id()  # Raises an error if Dark Souls III is not open

    def _load_env_args(self) -> Namespace:
        """Load the configuration parameters for the environment.

        Returns:
            The arguments as a Namespace object.
        """
        with open(self.config_path / (self.ENV_ID + ".yaml")) as f:
            return Namespace(**(yaml.load(f, Loader=yaml.SafeLoader)))

    def _set_game_properties(self):
        """Set general game properties that help gym stability"""
        self.game.lock_on_range = 50  # Increase lock on range for bosses
        self.game.los_lock_on_deactivate_time = 99  # Increase line of sight lock on deactivate time
        self.game.last_bonfire = self.env_args.bonfire
        self.game.player_stats = player_stats[self.ENV_ID]
        self.game.allow_moves = True
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_deaths = False
        self.game.allow_weapon_durability_dmg = False  # Weapons mustn't break during long sessions

    def _apply_action(self, action: int):
        """Apply an action to the environment.

        If the player is currently in an animation where he is disabled we omit all actions. The
        game queues actions and performs them as soon as the player is able to move. If an agent
        takes action 1 during the first step while being disabled, action 1 might be executed during
        the next step even though the agent has chosen action 2. In particular, roll and hit
        commands overwrite any normal movement. As long as we can't clear this queue we have to
        ensure actions are only performed when possible. Since we do not have access to the disabled
        game flags for the player we measure the duration of the current animation and compare that
        to the experimentally determined timeframe for this animation during which the player is
        disabled.

        Args:
            action: The action that is applied during this step.
        """
        player_animation = self._internal_state.player_animation
        player_animation_count = self._internal_state.player_animation_count
        if player_animation_count >= player_animations["standard"].get(player_animation, 1):
            self._game_input.update(actions[action])
        else:
            self._game_input.reset()

    def _step(self):
        """Perform the actual step ingame.

        Takes a 0.1s step ingame, handles critical events, updates the internal state and resets the
        player and boss HP.
        """
        self.game.resume_game()
        time.sleep(self._step_size)
        self.game.pause_game()
        game_state = self._game_logger.log()
        if not self._step_check(game_state):
            self._handle_critical_game_state(game_state)
            return
        self._update_internal_game_state(game_state)
        self.game.reset_player_hp()
        self.game.reset_boss_hp(self.ENV_ID)

    def _step_check(self, game_state: GameState) -> bool:
        """Check if game and player state are within expected values.

        Args:
            game_state: The current game state.

        Returns:
            True if the check passed, else False.

        Raises:
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        if game_state.player_hp == 0:
            logger.error("_step_check: Player HP is 0")
            logger.error(game_state)
            raise InvalidPlayerStateError("Player HP is 0")
        # Check if player is inside the borders of the arena
        bounds = (low < pos < high
                  for low, pos, high in zip(self.env_args.space_coords_low, game_state.
                                            player_pose[:2], self.env_args.space_coords_high))
        if not all(bounds):
            logger.error("_step_check: Player outside of arena bounds")
            logger.error(game_state)
            raise InvalidPlayerStateError("Player outside of arena bounds")
        # Critical animations need special recovery routines
        if game_state.player_animation in player_animations["critical"]:
            return False
        # Fall detection by lower state space bound on z coordinate
        if self.env_args.space_coords_low[2] > game_state.player_pose[2]:
            return False
        # During grab attacks, the lock cannot be established
        if not game_state.lock_on and game_state.player_animation not in ("ThrowAtk", "ThrowDef"):
            logger.debug("_step_check: Missing lock on detected")
            self._lock_on()
        # Unknown player animation. Shouldn't happen, add animation to tables!
        if game_state.player_animation not in player_animations["all"]:
            logger.warning(f"_step: Unknown player animation {game_state.player_animation}")
        if game_state.boss_animation not in boss_animations[self.ENV_ID]["all"]:
            logger.warning(f"_step: Unknown boss animation {game_state.boss_animation}")
        return True

    def _update_internal_game_state(self, game_state: GameState):
        """Update the internal game state.

        Args:
            game_state: The current game state.

        Raises:
            ResetNeeded: Tried to update before resetting the environment first.
        """
        if self._internal_state is None or self.done:
            logger.error("_update_internal_state: SoulsEnv.step() called before reset")
            raise ResetNeeded("SoulsEnv.step() called before reset")
        # Save animation counts and HP
        if self._internal_state.boss_animation == game_state.boss_animation:
            boss_animation_count = self._internal_state.boss_animation_count + 1
        else:
            boss_animation_count = 1
        if self._internal_state.player_animation == game_state.player_animation:
            player_animation_count = self._internal_state.player_animation_count + 1
        else:
            player_animation_count = 1
        player_hp, boss_hp = self._internal_state.player_hp, self._internal_state.boss_hp
        # Update animation count and HP
        self._internal_state = game_state
        self._internal_state.player_animation_count = player_animation_count
        self._internal_state.boss_animation_count = boss_animation_count
        self._internal_state.player_hp -= game_state.player_max_hp - player_hp
        self._internal_state.boss_hp -= game_state.boss_max_hp - boss_hp
        if self._internal_state.player_hp < 0:
            self._internal_state.player_hp = 0
        if self._internal_state.boss_hp < 0:
            self._internal_state.boss_hp = 0
        self.done = self._internal_state.player_hp == 0 or self._internal_state.boss_hp == 0

    def _handle_critical_game_state(self, game_state: GameState):
        """Handle critical game states.

        Args:
            game_state: The critical game state.
        """
        # Player is falling. Set player log HP to 0 and eagerly reset to prevent reload
        if game_state.player_pose[2] < self.env_args.space_coords_low[2]:
            game_state.player_hp = 0
            self._update_internal_state(game_state)
            self.game.reset_player_hp()
            self.game.reset_boss_hp(self.ENV_ID)
            self.game.player_pose = coordinates[self.ENV_ID]["player_init_pose"]
        if game_state.player_animation in player_animations["critical"]:
            game_state.player_hp = 0
            self._update_internal_state(game_state)

    def _lock_on(self, target_pose: Optional[np.ndarray] = None):
        """Reestablish lock on by orienting the camera towards the boss and pressing lock on.

        If the optional target pose is given, the camera is instead oriented towards the coordinates
        of the target pose. If the lock on attempt fails the camera is still oriented towards the
        target pose. This way we imitate a lock on even if we can't actually establish lock on.

        Note:
            Lock on is disabled during the *ThrowAtk* and *ThrowDef* player animations because it is
            disabled on the game side for these animations.

        Args:
            target_pose: The target pose towards which the camera should be oriented from its
                current position.
        """
        game_speed = self.game.global_speed
        self.game.pause_game()
        # During grab attacks, the lock cannot be established
        if self.game.player_animation not in ("ThrowAtk", "ThrowDef"):
            # Additional safeguard to make sure the player is currently not locked on
            if not self.game.lock_on:
                if target_pose is None:
                    target_pose = self.game.get_boss_pose(self.ENV_ID)
                self.game.camera_pose = target_pose[:3] - self.game.player_pose[:3]
                self._game_input.single_action("lockon")
                time.sleep(0.01)
                if not self.game.lock_on:
                    logger.debug("_lock_on: Failed to reestablish lock on")
                    # If the player is still oriented towards Iudex we essentially recover a lock on
                    # behavior. Pressing lock on turns the camera towards the player orientation. We
                    # therefore turn the camera towards Iudex again and continue without lock on
                    self.game.camera_pose = target_pose[:3] - self.game.player_pose[:3]
        self.game.global_speed = game_speed
