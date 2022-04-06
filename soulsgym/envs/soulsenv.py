"""Core abstract SoulsEnv class for gym environments in Dark Souls III."""
import time
import logging
from typing import Tuple, TypeVar, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from argparse import Namespace

import gym
import yaml
import numpy as np

from soulsgym.envs.utils.game_input import GameInput
from soulsgym.envs.utils.logger import Logger, GameState
from soulsgym.envs.utils.game_interface import Game
from soulsgym.envs.utils.static import coordinates, actions, player_animations, player_stats
from soulsgym.envs.utils.static import boss_animations
from soulsgym.envs.utils.game_window import GameWindow
from soulsgym.exception import LockOnFailure, ResetNeeded, InvalidPlayerStateError
from soulsgym.exception import InvalidBossStateError

logger = logging.getLogger(__name__)
ObsType = TypeVar("ObsType")


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
        self._game_logger = Logger()
        self._game_input = GameInput()
        self._game_window = GameWindow()
        self._check_ds3_running()
        self.game = Game()
        self.game.lock_on_range = 50  # Increase lock on range for bosses
        self.game.los_lock_on_deactivate_time = 99  # Increase line of sight lock on deactivate time
        self.game.resume_game()  # In case gym crashed while paused
        self.config_path = Path(__file__).parent / "config"
        self.env_args = self._load_env_args()
        self.game.player_stats = player_stats[self.ENV_ID]
        logger.info(self.env_args.init_msg)
        self._env_setup()
        logger.debug("Env init complete")
        # TODO: REMOVE
        self.unknown_boss_animations = []
        self.unknown_player_animations = []

    @abstractmethod
    def reset(self) -> ObsType:
        """Reset the environment to the beginning of an episode.

        Returns:
            The first observation after a reset.
        """

    @abstractmethod
    def _env_setup(self):
        """Execute the setup sequence for the boss fight."""

    @abstractmethod
    def compute_reward(self, obs: ObsType) -> float:  # TODO: Change to batch capable?
        """Compute the reward from a game observation.

        Args:
            obs: A game observation.

        Returns:
            The reward for the provided game observation.
        """

    def _check_ds3_running(self):
        """Check if the game is currently running."""
        self._game_window._get_ds_app_id()  # Raises an error if Dark Souls III is not open

    def _load_env_args(self) -> dict:
        """Load the configuration parameters for the environment.

        Returns:
            The arguments as a Namespace object.
        """
        with open(self.config_path / (self.ENV_ID + ".yaml")) as f:
            return Namespace(**(yaml.load(f, Loader=yaml.SafeLoader)))

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
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
        self._game_input.update(actions[action])
        self._step()
        reward = self.compute_reward(self._internal_state)
        if self.done:
            logger.debug("step: Episode finished")
        return self._internal_state, reward, self.done, {}

    def _step(self):
        """Perform the actual step ingame.

        Takes a 0.1s step ingame and updates the environment.

        Raises:
            InvalidPlayerStateError: The player animation is not known.
        """
        self.game.resume_game()
        time.sleep(self._step_size)
        self.game.pause_game()
        log = self._game_logger.log()
        self._step_check(log)
        # Critical animations need special recovery routines
        if log.player_animation in player_animations["critical"]:
            self._handle_critical_animation(log.player_animation)
            return
        self._update_internal_state(log)
        self.game.reset_player_hp()
        self.game.reset_target_hp()
        # Unknown player animation. Shouldn't happen, add animation to tables!
        if (log.player_animation
                not in player_animations["all"]) and (log.player_animation
                                                      not in self.unknown_player_animations):
            logger.warning(f"_step: Unknown player animation {log.player_animation}")
            self.unknown_player_animations.append(log.player_animation)
            # raise InvalidPlayerStateError(f"Unknown player animation: {log.player_animation}")
        if (log.boss_animation not in boss_animations[self.ENV_ID]["all"]) and (
                log.boss_animation not in self.unknown_boss_animations):
            self.unknown_boss_animations.append(log.boss_animation)
            logger.warning(f"_step: Unknown boss animation {log.boss_animation}")
            # raise InvalidBossStateError(f"Unknown boss animation: {log.boss_animation}")

    def _step_check(self, game_log: GameState):
        """Check if game and player state are within expected values.

        Raises:
            LockOnFailure: Lock on was lost and could not be reestablished.
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        if not game_log.locked_on:
            # During grap attacks, the lock cannot be established
            if game_log.player_animation not in ("ThrowAtk", "ThrowDef"):
                logger.debug("_step_check: Missing lock on detected")
                while not self.game.get_locked_on():  # TODO: Implement max resets
                    self._lock_on()
                game_log.locked_on = True
                if not self.game.get_locked_on():
                    logger.error("_step_check: Failed to reestablish lock on")
                    logger.error(game_log)
                    raise LockOnFailure("Failed to reestablish lock on")
        if game_log.player_hp == 0:
            logger.error("_step_check: Player HP is 0")
            logger.error(game_log)
            raise InvalidPlayerStateError("Player HP is 0")
        # Check if player is inside the borders of the arena
        bounds = (low < pos < high for low, pos, high in zip(
            self.env_args.space_coords_low, game_log.player_pos, self.env_args.space_coords_high))
        if not all(bounds):
            logger.error("_step_check: Player outside of arena bounds")
            logger.error(game_log)
            raise InvalidPlayerStateError("Player outside of arena bounds")
        return

    def _update_internal_state(self, game_log: GameState):
        """Update the internal game state.

        Args:
            game_log: Current GameState from the game logger.

        Raises:
            ResetNeeded: Tried to update before resetting the environment first.
        """
        if self._internal_state is None or self.done:
            logger.error("_update_internal_state: Failed, SoulsEnv.step() called before reset")
            raise ResetNeeded("SoulsEnv.step() called before reset")
        # Save animation counts and HP
        if self._internal_state.boss_animation == game_log.boss_animation:
            boss_animation_count = self._internal_state.boss_animation_count + 1
        else:
            boss_animation_count = 0
        if self._internal_state.player_animation == game_log.player_animation:
            player_animation_count = self._internal_state.player_animation_count + 1
        else:
            player_animation_count = 0
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

    def _handle_critical_animation(self, p_anim: str):
        if "Fall" in p_anim:
            # Player is falling. Set 0 HP and rely on reset for teleport to prevent death
            log = self._game_logger.log()
            log.player_hp = 0
            self._update_internal_state(log)
            self.game.reset_player_hp()
            self.game.reset_target_hp()
            # Eagerly teleport player to safety so that the fall is guaranteed to be nonlethal
            self.game.player_position = coordinates[self.ENV_ID]["player_init_pos"]

    def close(self):
        """Unpause the game and kill the player to restore the original game state."""
        self.game.resume_game()
        self._game_input.reset()
        # Restore game parameter defaults
        self.game.lock_on_range = 15
        self.game.los_lock_on_deactivate_time = 2
        self.game.target_attacks = True
        self.game.reload()
        logger.debug("SoulsEnv close successful")

    def _lock_on(self, target_position: Optional[List[float]] = None):
        """Reestablish lock on by rotating the camera around the player and press lock on."""
        game_speed = self.game.global_speed
        self.game.pause_game()
        t_pos = target_position or self.game.target_position
        p_pos = self.game.player_position
        self.game.camera_pose = [t_pos[0] - p_pos[0], t_pos[1] - p_pos[1], t_pos[2] - p_pos[2]]
        self._game_input.single_action("lockon")
        self.game.global_speed = game_speed
