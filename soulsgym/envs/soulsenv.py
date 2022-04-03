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
from soulsgym.envs.utils.game_window import GameWindow
from soulsgym.exception import LockOnFailure, ResetNeeded, InvalidPlayerStateError

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
        self.game.resume_game()  # In case gym crashed while paused
        self.config_path = Path(__file__).parent / "config"
        self.env_args = self._load_env_args()
        self.game.player_stats = player_stats[self.ENV_ID]
        logger.info(self.env_args.init_msg)
        self._env_setup()
        logger.debug("Env init complete")

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
        self._step_callback()
        reward = self.compute_reward(self._internal_state)
        if self.done:
            logger.debug("step: Episode finished")
        return self._internal_state, reward, self.done, {}

    def _step_callback(self):
        """Perform the actual step ingame.

        Handles multisteps while being disabled.

        Raises:
            InvalidPlayerStateError: The player animation is not known.
        """
        self._sub_step()
        while not self.done:  # Take substeps until the player animation or episode is finished
            p_anim = self._internal_state["player_animation"]
            p_anim = p_anim if p_anim is not None else "None"
            # Uninterruptable animation, wait for finish and exit loop. Guaranteed to take x steps
            if p_anim in player_animations["no_interrupt"].keys():
                # First step already taken before the loop
                for _ in range(player_animations["no_interrupt"][p_anim] - 1):
                    self._sub_step()
                    if self.done:
                        break
                break
            # Interruptable animation. If animation changes during execution, start again with new
            # animation
            elif p_anim in player_animations["interrupt"].keys():
                anim_cnt = -1  # If range is 0, anim_cnt not assigned, but needs to be -1 to break
                for anim_cnt in range(player_animations["interrupt"][p_anim] - 1):
                    self._sub_step()
                    if self._internal_state.player_animation != p_anim or self.done:
                        break
                # Animation count had finished
                if anim_cnt == player_animations["interrupt"][p_anim] - 2:
                    break
            # Critical animations which need special recovery routines
            elif p_anim in player_animations["critical"]:
                self._handle_critical_animation(p_anim)
            # Unknown player animation. Shouldn't happen, add animation to tables!
            else:
                logger.error(f"_step_callback: Unknown player animation {p_anim}")
                raise InvalidPlayerStateError(f"Unknown player animation: {p_anim}")

    def _sub_step(self):
        """Perform a 0.1s step ingame and update the environment."""
        self.game.resume_game()
        time.sleep(self._step_size)
        self.game.pause_game()
        log = self._game_logger.log()
        self._sub_step_check(log)
        self._update_internal_state(log)
        self.game.reset_player_hp()
        self.game.reset_target_hp()

    def _sub_step_check(self, game_log: GameState):
        """Check if game and player state are within expected values.

        Raises:
            LockOnFailure: Lock on was lost and could not be reestablished.
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        if not game_log.locked_on:
            logger.debug("_sub_step_check: Missing lock on detected")
            # During grap attacks, the lock cannot be established
            if game_log.player_animation not in ("ThrowAtk", "ThrowDef"):
                res = self._lock_on()
                if not res:
                    logger.error("_sub_step_check: Failed to reestablish lock on")
                    raise LockOnFailure("Failed to reestablish lock on")
        if game_log.player_hp == 0:
            logger.debug("_sub_step_check: Player HP is 0")
            raise InvalidPlayerStateError("Player HP is 0")
        # Check if player is inside the borders of the arena
        limits = [coordinates[self.ENV_ID]["limits"][c] for c in ["x", "y", "z"]]
        if not all([lim[0] < pos < lim[1] for pos, lim in zip(game_log.player_pos, limits)]):
            logger.debug("_sub_step_check: Player outside of arena bounds")
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
        player_hp, boss_hp = self._internal_state.player_hp, self._internal_state.boss_hp
        self._internal_state = game_log
        self._internal_state.player_hp -= game_log.player_max_hp - player_hp
        self._internal_state.boss_hp -= game_log.boss_max_hp - boss_hp
        if self._internal_state.player_hp < 0:
            self._internal_state.player_hp = 0
        if self._internal_state.boss_hp < 0:
            self._internal_state.boss_hp = 0
        self.done = self._internal_state.player_hp == 0 or self._internal_state.boss_hp == 0

    def _handle_critical_animation(self, p_anim: str):
        if p_anim == "FallStart":
            # Player is falling. Set 0 HP and rely on reset for teleport to prevent death
            log = self._game_logger.log()
            self._sub_step_check(log)
            log.player_hp = 0
            self._update_internal_state(log)
            self.game.reset_player_hp()
            self.game.reset_target_hp()

    def close(self):
        """Unpause the game and kill the player to restore the original game state."""
        self.game.resume_game()
        self._game_input.reset()
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
