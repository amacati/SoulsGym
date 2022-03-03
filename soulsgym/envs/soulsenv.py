import time
import logging
from typing import Optional, Tuple, TypeVar
from pathlib import Path
from abc import ABC, abstractmethod
from argparse import Namespace

import gym
import yaml

from soulsgym.envs.utils.game_input import GameInput
from soulsgym.envs.utils.logger import Logger, GameState
import soulsgym.envs.utils.game_interface as game
import soulsgym.envs.utils.tables as tables
from soulsgym.envs.utils.tables import coordinates, actions, player_animations
from soulsgym.envs.utils.game_window import GameWindow
from soulsgym.exception import LockOnFailure, ResetNeeded, InvalidPlayerStateError

logger = logging.getLogger("SoulsGym")
ObsType = TypeVar("ObsType")


class SoulsEnv(gym.Env, gym.utils.EzPickle, ABC):
    metadata = {'render.modes': ['human']}
    ENV_ID = ""
    _step_size = 0.1

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(tables.action_list))
        self._internal_state = None
        self.done = False
        self._game_logger = Logger()
        self._game_input = GameInput()
        self._game_window = GameWindow()
        self._check_ds3_running()
        game.resume_game()  # In case gym crashed while paused
        self.config_path = Path(__file__).parent / "config"
        self.env_args = self._load_env_args()
        logger.info(self.env_args.init_msg)
        self._env_setup()
        logger.debug("Gym init complete")

    @abstractmethod
    def reset(self) -> ObsType:
        ...

    @abstractmethod
    def _env_setup(self):
        ...

    @abstractmethod
    def compute_reward(self):
        ...

    def _check_ds3_running(self):
        self._game_window._get_ds_app_id()

    def _load_env_args(self) -> dict:
        with open(self.config_path / (self.ENV_ID + ".yaml")) as f:
            return Namespace(**(yaml.load(f, Loader=yaml.SafeLoader)))

    def step(self, action):
        if self.done:
            logger.error("step: Environment step called after environment was done")
            raise ResetNeeded("Environment step called after environment was done")
        self._game_input.array_update(actions[action])
        self._step_callback()
        reward = self.compute_reward(self._internal_state)
        return self._internal_state, reward, self.done, {}

    def _step_callback(self):
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
            # Unknown player animation. Shouldn't happen, add animation to tables!
            else:
                logger.error(f"_step_callback: Unknown player animation {p_anim}")
                raise InvalidPlayerStateError(f"Unknown player animation: {p_anim}")

    def _sub_step(self):
        game.resume_game()
        time.sleep(self._step_size)
        game.pause_game()
        log = self._game_logger.log()
        self._sub_step_check(log)
        self._update_internal_state(log)
        game.reset_player_hp()
        game.reset_targeted_hp()

    def _sub_step_check(self, game_log):
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
        limits = [coordinates["iudex_limits"][c] for c in ["x", "y", "z"]]
        if not all([lim[0] < pos < lim[1] for pos, lim in zip(game_log.player_pos, limits)]):
            logger.debug("_sub_step_check: Player outside of arena bounds")
            raise InvalidPlayerStateError("Player outside of arena bounds")
        return

    def _update_internal_state(self, game_log):
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

    @staticmethod
    def close():
        game.resume_game()
        game.reset_iudex_and_die()  # TODO: Replace with generic death
        logger.debug("SoulsEnv close successful")

    def _lock_on(self) -> bool:
        for d in range(5):
            self._game_input.single_action("CameraRight", press_time=.5 * d)  # Spin camera around
            self._game_input.single_action("LockOn")
            time.sleep(
                0.5)  # Give some time to either reset cam or propagate lock on. TODO: Optimize time
            if game.get_locked_on():
                return True
        for d in range(1, 5):
            self._game_input.single_action("CameraLeft", press_time=.5 * d)  # Walk into the arena
            self._game_input.single_action("LockOn")
            time.sleep(0.5)  # Give some time to either reset cam or propagate lock on
            if game.get_locked_on():
                return True
        return False


class SoulsGym:
    _UNALTERED_STATES = [
        "player_sp", "boss_max_hp", "iudex_def", "player_x", "player_y", "player_z", "player_a",
        "boss_x", "boss_y", "boss_z", "boss_a", "phase2", "animation", "cam_lock",
        "animation_count", "player_animation"
    ]

    actions = actions
    MAX_REWARD = 1
    MIN_REWARD = -1.05
    STATE_DIM = GameState.state_size()
    ACTION_DIM = len(tables.action_list)
    action_space = gym.spaces.Discrete(len(tables.action_list))

    def __init__(self):
        """
        Initializes the game input manager, the logger and the game window manager.
        """
        self.state = None
        self.done = False
        self._step_size = 0.1  # Changing this requires redo of player animation duration in tables!
        self._game_input = GameInput()
        self._ds_logger = Logger()
        self._game_window = GameWindow()
        self.logger = logging.getLogger("SoulsGym")

        self._restart_required = False  # Restarts the Gym if player has died during an episode
        self.logger.debug("Gym init complete")

    def create(self, init_retries: int = 3) -> Tuple[GameState, float, bool]:
        """
        Creates the Gym environment.

        Returns:
            The current state of the gym, the reward and the episode finish flag.
        """
        while not self._post_init_health_check() and init_retries > 0:
            self._init_ds_env()
            self._initial_key_press()
            init_retries -= 1
            if not init_retries:
                self.logger.error("Maximum number of teleport resets exceeded")
                raise RuntimeError("Maximum number of teleport resets exceeded")
        self.logger.debug("Gym start state reached")
        game.pause_game()
        self.state = self._ds_logger.log()
        self.done = False
        self._restart_required = False
        return self.state, 0, self.done

    def step(self, action: Optional[int] = None) -> Tuple[GameState, float, bool]:
        """
        Performs a single gym step.

        Resumes the game for ``self._step_size`` seconds, pauses, logs the game state, checks for
        errors, updates the internal state gym state and resets the boss and player hp.

        Args:
            action: The index of the game input action that should be performed as key-presses.

        Returns:
            The current state of the gym, the reward and the episode finish flag.
        """
        if self._restart_required:
            self.logger.error("Gym.step() called with restart required")
            raise RuntimeError("Gym episode required restart but has continued with step.")
        if action:
            self._game_input.array_update(self.actions[action])
            self.logger.debug("Game input registered")
        self._sub_step()
        while not self.done:  # Take substeps until the player animation or episode is finished
            p_anim = self.state["player_animation"]
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
                    if self.state.player_animation != p_anim or self.done:
                        break
                # Animation count had finished
                if anim_cnt == player_animations["interrupt"][p_anim] - 2:
                    break
            # Unknown player animation. Shouldn't happen, add animation to tables!
            else:
                self.logger.error(f"Unknown player animation {p_anim} in Gym.step()")
                raise RuntimeError(f"Unknown player animation: {p_anim}")
        reward = self._compute_reward()
        return self.state, reward, self.done

    def _sub_step(self):
        game.resume_game()
        time.sleep(self._step_size)
        game.pause_game()
        log = self._ds_logger.log()
        self._check_health(log)
        self._update_state(log)
        self._reset_health()

    def reset(self) -> Tuple[GameState, float, bool]:
        """
        Resets the gym to the start of a training episode state.

        Returns:
            The current state of the gym, the reward and the episode finish flag.
        """
        if self._restart_required:  # Player has died, gym needs to be recreated.
            self.logger.debug("Restart required for reset, recreating Gym")
            self.create()
            self.logger.debug("Successfully created Gym")
            self._restart_required = False
        self.state = self._ds_logger.log()
        self.done = False
        return self.state, 0, self.done

    def soft_reset(self) -> Tuple[GameState, float, bool]:
        """
        Resets the internal gym state, but does not reset player position and boss position.

        Useful because resetting the complete gym takes a lot of time (reloading, teleporting, ...).
        Use in continuous training sample generation.

        TODO: For better distribution of data, we should teleport both the player and Iudex to their
        starting position.

        Returns:
            The current state of the gym, the reward and the episode finish flag.
        """
        if self._restart_required:
            self._restart_required = False
        self._reset_health()
        self.state = self._ds_logger.log()
        self.done = False
        return self.state, 0., self.done

    def close(self) -> None:
        """
        Safely closes the gym and resets the game settings.
        """
        game.resume_game()  # In case the game was paused when calling close
        game.reset_iudex_and_die()
        self.logger.debug("Reset and death successful")
        self.state = None
        self.done = False
        self._restart_required = False
        self.logger.debug("Gym closed")

    def _init_ds_env(self) -> None:
        """
        Sets all game variables for the gym, resets by killing the player and teleports him to the
        arena entrance.
        """
        game.resume_game()  # In case Dark Souls crashed with global time speed to 0
        game.reset_iudex_and_die()  # Initializes the environment for the boss.
        self.logger.debug("Reset and death successful")
        time.sleep(3)  # Wait until the loading screen comes up to click into the application
        self._game_window.focus_application()
        self.logger.debug("Application focus successful. Focus change by intervention possible")
        game.clear_cache()  # Reset memory manipulator cache after death has invalidated addresses
        time.sleep(10)  # Wait for invalid values of animation to ceise
        loaded = False
        while not loaded:
            time.sleep(1)  # Give the game time to load.
            loaded = self._ds_logger.log(no_target=True).player_animation == "Idle"
        self.logger.debug("Player respawn registered")
        game.teleport_player(coordinates['iudex'])
        time.sleep(1)
        self.logger.debug("Player teleport successful")

    def _initial_key_press(self) -> None:
        """
        Performs the inital key presses for entering the arena and locking onto Gundyr.
        """
        self._game_input.single_action("Interact")  # Enters boss fog wall
        time.sleep(3)  # Await fog animation finish
        self._game_input.single_action("Forward", press_time=4.0)  # Walk into the arena
        self._game_input.single_action("LockOn")  # Lock onto Gundyr
        time.sleep(0.5)  # Give some time to propagate the locked-on state
        self.logger.debug("Inital key presses executed")

    def _post_init_health_check(self):
        """
        Performs a basic health check to see if initalization was successful.
        """
        #  Check if iudex is already defeated
        if game.get_iudex_defeated() or not game.get_iudex_encountered():
            self.logger.debug("Health check failed: Boss flags not set properly")
            return False

        # Check if position is okay.
        p_pos = game.get_player_position()
        dst = sum([(tables.coordinates["player_init_pos"][i] - p_pos[i])**2 for i in range(3)])**.5
        if dst > 3:
            self.logger.debug("Init health check failed: Post keypress position out of tolerances")
            return False

        # Check if player didn't die yet.
        p_hp, _ = game.get_player_hp_sp()
        if p_hp == 0:
            self.logger.debug("Init health check failed: Player HP is 0")
            return False

        # Check if iudex is locked on.
        if not game.get_locked_on():
            self.logger.debug("Init health check failed: No lock on detected")
            return False

        self.logger.debug("Init health check successful")
        return True

    def _check_health(self, log: GameState) -> None:
        """
        Checks if current log is within expected values (no cliff death etc).

        Raises:
            RuntimeError: If the gym encounters log states outside the expected range.
        """
        if not log.cam_lock:
            self.logger.debug("Health check: Missing lock on detected")
            # During grap attacks, the lock cannot be established
            if log.player_animation not in ("ThrowAtk", "ThrowDef"):
                success = self._try_reestablish_lockon()
                self._restart_required = not success
                if success:
                    self.logger.debug("Health check: Lock on restored")
                else:
                    self.logger.debug("Health check: Lock on restore failed, restart signaled")
        if log.player_hp == 0:
            self.logger.debug("Health check: Player HP is 0")
            self._restart_required = True

        # Check if player is inside the borders of the arena
        for (pos, lim) in zip([log.player_x, log.player_y, log.player_z],
                              [tables.coordinates["iudex_limits"][c] for c in ["x", "y", "z"]]):
            if pos < lim[0] or pos > lim[1]:
                self.logger.debug("Health check: Player outside of arena bounds. Restart signaled")
                self._restart_required = True
                break

    def _update_state(self, log: GameState) -> None:
        """
        Updates the internal gym state.

        The gym maintains an internal state to keep track of the total damage dealt to the player
        and the boss. We reset the ingame health for both to prevent deaths and the associated
        loading times, therefore we can't use the log directly.

        Args:
            log: The current game log.
        """
        if self.state is None or self.done:
            self.logger.error("State update failed, Gym.step() called before create or reset")
            raise RuntimeError("Gym.step() called before create or reset")
        for key in self._UNALTERED_STATES:
            self.state[key] = log[key]
        self.state.player_hp -= log.player_max_hp - log.player_hp
        self.state.boss_hp -= log.boss_max_hp - log.boss_hp
        if self.state.player_hp < 0:
            self.state.player_hp = 0
        if self.state.boss_hp < 0:
            self.state.boss_hp = 0
        self.done = self.state.player_hp == 0 or self.state.boss_hp == 0

    def _reset_health(self) -> None:
        """
        Resets both player and boss hp.
        """
        game.reset_player_hp()
        game.reset_targeted_hp()

    def _try_reestablish_lockon(self) -> bool:
        for d in range(5):
            self._game_input.single_action("CameraRight", press_time=.5 * d)  # Walk into the arena
            self._game_input.single_action("LockOn")
            time.sleep(0.5)  # give some time to either reset cam or propagate lock on
            if game.get_locked_on():
                return True

        for d in range(1, 5):
            self._game_input.single_action("CameraLeft", press_time=.5 * d)  # Walk into the arena
            self._game_input.single_action("LockOn")
            time.sleep(0.5)  # give some time to either reset cam or propagate lock on
            if game.get_locked_on():
                return True

        return False

    def _compute_reward(self) -> float:
        """
        Computes the reward for the current game state.

        Returns:
            The reward.
        """
        player_hp_penalty = 1 - self.state.player_hp / self.state.player_max_hp
        player_sp_penalty = 1 - self.state.player_sp / self.state.player_max_sp
        boss_hp_reward = 1 - self.state.boss_hp / self.state.boss_max_hp
        return boss_hp_reward - player_hp_penalty - 0.05 * player_sp_penalty
