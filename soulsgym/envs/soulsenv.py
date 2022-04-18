"""Core abstract SoulsEnv class for gym environments in Dark Souls III."""
import time
import logging
from typing import Tuple, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import deque

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
    """Abstract base class for Dark Souls III gym environments.

    The environment keeps track of the player's and boss' HP/SP and resets the ingame values
    periodically. This avoids costly reloads because the player does not actually die at the end of
    an episode.
    """

    metadata = {'render.modes': ['human']}
    ENV_ID = ""  # Each SoulsGym has to define its own ID and name the config files accordingly
    _step_size = 0.1

    def __init__(self):
        """Initialize the game managers and run the environment setup sequence."""
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
        self.game.lock_on_range = 50  # Increase lock on range for bosses
        self.game.los_lock_on_deactivate_time = 99  # Increase line of sight lock on deactivate time
        self.gravity = True
        self.game.resume_game()  # In case gym crashed while paused
        self.config_path = Path(__file__).parent / "config"
        self.env_args = self._load_env_args()
        self.game.player_stats = player_stats[self.ENV_ID]
        logger.info(self.env_args.init_msg)
        self._env_setup()
        self.game.pause_game()
        logger.debug("Env init complete")
        # TODO: REMOVE
        self.img_cache = deque(maxlen=20)

    @abstractmethod
    def reset(self) -> GameState:
        """Reset the environment to the beginning of an episode.

        Returns:
            The first observation after a reset.
        """

    @abstractmethod
    def _env_setup(self):
        """Execute the setup sequence for the boss fight."""

    @abstractmethod
    def compute_reward(self, obs: GameState) -> float:  # TODO: Change to batch capable?
        """Compute the reward from a game observation.

        Args:
            obs: A game observation.

        Returns:
            The reward for the provided game observation.
        """

    def _game_check(self):
        """Check if the game is currently running."""
        self._game_window._get_ds_app_id()  # Raises an error if Dark Souls III is not open

    def _load_env_args(self) -> dict:
        """Load the configuration parameters for the environment.

        Returns:
            The arguments as a Namespace object.
        """
        with open(self.config_path / (self.ENV_ID + ".yaml")) as f:
            return Namespace(**(yaml.load(f, Loader=yaml.SafeLoader)))

    def step(self, action: int) -> Tuple[GameState, float, bool, dict]:
        """Perform a step forward in the environment with a given action.

        Environment step size is 0.1s. The game is halted before and after the step. If the player
        is disabled, the environment steps forward (with 0.1s steps) until the player is able to act
        again.

        Args:
            action: The action that is applied during this step.

        Returns:
            A tuple of the next observed state, the reward, the done flag and additional info.

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

    def _apply_action(self, action: int):
        player_animation = self._internal_state.player_animation
        player_animation_count = self._internal_state.player_animation_count
        if player_animation_count >= player_animations["standard"].get(player_animation, 1):
            self._game_input.update(actions[action])
        else:
            self._game_input.reset()

    def _step(self):
        """Perform the actual step ingame.

        Takes a 0.1s step ingame and updates the environment.

        Raises:
            InvalidPlayerStateError: The player animation is not known.
        """
        self.game.resume_game()
        time.sleep(self._step_size)
        self.game.pause_game()
        # TODO: REMOVE
        # self.img_cache.append(self._game_window.screenshot())
        # END TODO
        log = self._game_logger.log()
        if not self._step_check(log):
            self._handle_critical_log(log)
            return
        self._update_internal_state(log)
        self.game.reset_player_hp()
        self.game.reset_boss_hp(self.ENV_ID)

    def _step_check(self, game_log: GameState) -> bool:
        """Check if game and player state are within expected values.

        Returns:
            True if the check passed, else False.

        Raises:
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        # During grab attacks, the lock cannot be established
        if not game_log.lock_on and game_log.player_animation not in ("ThrowAtk", "ThrowDef"):
            logger.debug("_step_check: Missing lock on detected")
            self._lock_on()
        if game_log.player_hp == 0:
            logger.error("_step_check: Player HP is 0")
            logger.error(game_log)
            raise InvalidPlayerStateError("Player HP is 0")
        # Check if player is inside the borders of the arena
        bounds = (low < pos < high
                  for low, pos, high in zip(self.env_args.space_coords_low, game_log.
                                            player_pose[:2], self.env_args.space_coords_high))
        if not all(bounds):
            logger.error("_step_check: Player outside of arena bounds")
            logger.error(game_log)
            raise InvalidPlayerStateError("Player outside of arena bounds")
        # Critical animations need special recovery routines
        if game_log.player_animation in player_animations["critical"]:
            return False
        # Fall detection by lower state space bound on z coordinate
        if self.env_args.space_coords_low[2] > game_log.player_pose[2]:
            return False
        # Unknown player animation. Shouldn't happen, add animation to tables!
        if game_log.player_animation not in player_animations["all"]:
            logger.warning(f"_step: Unknown player animation {game_log.player_animation}")
        if game_log.boss_animation not in boss_animations[self.ENV_ID]["all"]:
            logger.warning(f"_step: Unknown boss animation {game_log.boss_animation}")
        return True

    def _update_internal_state(self, game_log: GameState):
        """Update the internal game state.

        Args:
            game_log: Current GameState from the game logger.

        Raises:
            ResetNeeded: Tried to update before resetting the environment first.
        """
        if self._internal_state is None or self.done:
            logger.error("_update_internal_state: SoulsEnv.step() called before reset")
            raise ResetNeeded("SoulsEnv.step() called before reset")
        # Save animation counts and HP
        if self._internal_state.boss_animation == game_log.boss_animation:
            boss_animation_count = self._internal_state.boss_animation_count + 1
        else:
            boss_animation_count = 1
        if self._internal_state.player_animation == game_log.player_animation:
            player_animation_count = self._internal_state.player_animation_count + 1
        else:
            player_animation_count = 1
        player_hp, boss_hp = self._internal_state.player_hp, self._internal_state.boss_hp
        # Update animation count and HP
        self._internal_state = game_log
        self._internal_state.player_animation_count = player_animation_count
        self._internal_state.boss_animation_count = boss_animation_count
        self._internal_state.player_hp -= game_log.player_max_hp - player_hp
        self._internal_state.boss_hp -= game_log.boss_max_hp - boss_hp
        if self._internal_state.player_hp < 0:
            self._internal_state.player_hp = 0
        if self._internal_state.boss_hp < 0:
            self._internal_state.boss_hp = 0
        self.done = self._internal_state.player_hp == 0 or self._internal_state.boss_hp == 0

    def _handle_critical_log(self, log: GameState):
        # Player is falling. Set player log HP to 0 and eagerly reset to prevent reload
        if log.player_pose[2] < self.env_args.space_coords_low[2]:
            log.player_hp = 0
            self._update_internal_state(log)
            self.game.reset_player_hp()
            self.game.reset_boss_hp(self.ENV_ID)
            self.game.player_pose = coordinates[self.ENV_ID]["player_init_pose"]
        if log.player_animation in player_animations["critical"]:
            log.player_hp = 0
            self._update_internal_state(log)

    def close(self):
        """Unpause the game and kill the player to restore the original game state."""
        self.game.resume_game()
        self._game_input.reset()
        # Restore game parameter defaults
        self.game.lock_on_range = 15
        self.game.los_lock_on_deactivate_time = 2
        self.gravity = True
        self.game.set_boss_attacks(self.ENV_ID, True)
        self.game.player_hp = 0  # Kill player to force game reload. Don't wait for completion
        logger.debug("SoulsEnv close successful")

    def _lock_on(self, target_pose: Optional[np.ndarray] = None):
        """Reestablish lock on by rotating the camera around the player and press lock on."""
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
                    logger.warning("_lock_on: Failed to reestablish lock on")
                    # If the player is still oriented towards Iudex we essentially recover a lock on
                    # behavior. Pressing lock on turns the camera towards the player orientation. We
                    # therefore turn the camera towards Iudex again and continue without lock on
                    self.game.camera_pose = target_pose[:3] - self.game.player_pose[:3]
        self.game.global_speed = game_speed
