"""The ``soulsenv`` module provides the abstract base class for all ``soulsgym`` environments.

This :class:`~.SoulsEnv` class includes the general gym logic and defines abstract methods that all
environments have to implement.

In addition, we also provide a :class:`~.SoulsEnvDemo` base class for demo environments. In contrast
to the training environments, demos cover all phases of a boss fight and allow to demonstrate the
agent's abilities in a setting that is as close to the real game as possible.
"""
import logging
from typing import Tuple, List, Any, Dict
from pathlib import Path
from abc import ABC, abstractmethod
from argparse import Namespace

import gymnasium
import yaml
import numpy as np

from soulsgym.core.game_input import GameInput
from soulsgym.core.game_state import GameState
from soulsgym.core.games import game_factory
from soulsgym.core.static import coordinates, actions, player_animations, player_stats
from soulsgym.core.static import critical_player_animations, boss_animations
from soulsgym.core.utils import get_pid, wrap_to_pi
from soulsgym.core.game_window import GameWindow
from soulsgym.exception import GameStateError, ResetNeeded, InvalidPlayerStateError

logger = logging.getLogger(__name__)


class SoulsEnv(gymnasium.Env, ABC):
    """Abstract base class for ``soulsgym`` environments.

    Each ``SoulsEnv`` initializes a :class:`.GameInput`, a :class:`.GameWindow`, a :class:`.Game`
    interface and a :class:`.Logger` to read from and to the game. During an episode the game is
    paused per default. At each :meth:`.SoulsEnv.step` call the environment applies the current
    input if valid (see :meth:`.SoulsEnv._apply_action` for details). It then unpauses the game,
    waits for ``step_size`` seconds and pauses again. During that time, the game runs with a speed
    multiplier defined by ``game_speed``. The new game state is logged and processed to update the
    internal game state. The environment tries to detect and recover from any unexpected errors.

    To solve any camera control issues we lock on to the boss at all times. For exceptions see
    :meth:`.SoulsEnv._lock_on`.

    Note:
        We deem it too complicated to learn from the image alone. The :class:`.GameWindow` class is
        however designed to provide this capability and the gym can easily be extended to yield
        image data as well.

    Warning:
        The target game has to be running at the initialization of :class:`SoulsEnv`.

    Warning:
        Setting ``game_speed`` too high might result in unstable behaviour. The maximal value is
        determined by hardware factors. We strongly recommend to stick to values in the range of
        [1, 3].
    """

    metadata = {'render_modes': []}
    ENV_ID = ""  # Each SoulsGym has to define its own ID and name the config files accordingly
    step_size = 0.1

    def __init__(self, game_speed: int = 1.):
        """Initialize the game managers, load the environment config and set the game properties.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
        """
        super().__init__()
        assert game_speed > 0, "Game speed must be positive!"
        self._game_check()  # Check if the game is running, otherwise we can't initialize
        # Initialize game managers
        self.game = game_factory(self.game_id)
        self._game_input = GameInput()
        self._game_window = GameWindow(self.game_id)
        # Check if the player has loaded into the game
        if not self.game.is_ingame:
            logger.error("Player is not loaded into the game")
            raise GameStateError("Player is not loaded into the game")
        # Initialize helper variables
        self._game_speed = game_speed
        self._internal_state = None
        self._last_player_animation_time = 0
        self._last_boss_animation_time = 0
        self._lock_on_timer = 0  # Steps until "lock on" press is allowed
        self._is_init = False
        self.terminated = False
        # Load environment config
        self.config_path = Path(__file__).parent / "config"
        self.env_args = self._load_env_args()
        self._set_game_properties()
        logger.info(self.env_args.init_msg)
        logger.debug("Env init complete")

    @property
    @abstractmethod
    def game_id(self):
        """Every Souls game has to define the base game (e.g. DarkSoulsIII, EldenRing, ...)."""

    @abstractmethod
    def reset(self, seed: int | None = None, options: Any | None = None) -> Tuple[dict, dict]:
        """Reset the environment to the beginning of an episode.

        Args:
            seed: Random seed. Required by gymnasium, but does not apply to SoulsGyms.
            options: Options argument required by gymnasium. Not used in SoulsGym.

        Returns:
            A tuple of the first game state and the info dict after the reset.
        """

    @abstractmethod
    def _env_setup(self):
        """Execute the setup sequence for the boss fight."""

    @staticmethod
    @abstractmethod
    def compute_reward(game_state: GameState, next_game_state: GameState) -> float:
        """Compute the reward from the current game state and the next game state.

        Args:
            game_state: The game state before the step.
            next_game_state: The game state after the step.

        Returns:
            The reward for the provided game states.
        """

    @property
    @abstractmethod
    def obs(self) -> Dict:
        """Return the current observation of the environment."""

    @property
    @abstractmethod
    def info(self) -> Dict:
        """Return the current info dict of the environment."""

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """Perform a step forward in the environment with a given action.

        Each step advances the ingame time by `step_size` seconds. The game is paused before and
        after the step.

        Args:
            action: The action that is applied during this step.

        Returns:
            A tuple of the next game state, the reward, the terminated flag, the truncated flag, and
            an additional info dictionary.

        Raises:
            ResetNeeded: `step()` was called after the episode was already finished.
        """
        if self.terminated:
            logger.error("Environment step called after environment was terminated")
            raise ResetNeeded("Environment step called after environment was terminated")
        previous_game_state = self._internal_state.copy()
        self._step(action)
        reward = self.compute_reward(previous_game_state, self._internal_state)
        if self.terminated:
            logger.debug("Episode finished")
        return self.obs, reward, self.terminated, False, self.info

    def close(self):
        """Unpause the game, reset altered game properties and reload.

        Note:
            Does not wait for reload to complete.
        """
        self.game.resume_game()
        self._game_input.reset()
        # Restore game parameter defaults
        self.game.lock_on_bonus_range = 0
        self.game.los_lock_on_deactivate_time = 2
        self.game.allow_moves = True
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_deaths = True
        self.game.allow_weapon_durability_dmg = True
        self.game.player_hp = 0  # Kill player to force game reload. Don't wait for completion
        logger.debug("SoulsEnv close successful")

    def current_valid_actions(self) -> List[int]:
        """Get the set of currently valid actions.

        Returns:
            An array of integers containing the currently allowed actions.
        """
        if self._internal_state is None:
            return []
        player_animation = self._internal_state.player_animation
        durations = player_animations.get(player_animation, {"timings": [0., 0., 0.]})["timings"]
        current_duration = self._internal_state.player_animation_duration
        player_sp = self._internal_state.player_sp
        # Movement actions (duration index 2) do not require SP
        movement_ids = list(range(8)) if current_duration >= durations[2] else []
        # Roll actions (duration index 1) require SP > 0
        roll_ids = list(range(8, 16)) if player_sp > 0 and current_duration >= durations[1] else []
        # Hit actions and parry (duration index 0) require SP > 0
        attack_ids = [16, 17, 18] if player_sp > 0 and current_duration >= durations[0] else []
        # ID 19 (do nothing) is always a valid action
        movement_ids + roll_ids + attack_ids + [19]
        return movement_ids + roll_ids + attack_ids + [19]

    def seed(seed: Any) -> List[int]:
        """Set the random seed for the environment.

        Since we cannot control the randomness of the game and can't precisely control the game,
        loop, this function does not have any effect.

        Note:
            Setting the seed will **not** lead to reproducible results!

        Args:
            seed: Random seed.

        Returns:
            A list with 0 to comply with OpenAI's function signature.
        """
        logger.warning("Trying to set the seed, but SoulsGym can't control randomness in the game")
        return [0]

    def render(self):
        """Render the environment.

        This is a no-op since we can't render the environment and the game has to be open anyways.
        """
        logger.warning("Rendering the environment is useless, game has to be open anyways.")

    def _game_check(self):
        """Check if the game is currently running."""
        get_pid(self.game_id)  # Raises an error if the game is not open

    def _load_env_args(self) -> Namespace:
        """Load the configuration parameters for the environment.

        Returns:
            The arguments as a Namespace object.
        """
        with open(self.config_path / (self.ENV_ID + ".yaml")) as f:
            return Namespace(**(yaml.load(f, Loader=yaml.SafeLoader)))

    def _set_game_properties(self):
        """Set general game properties that help gym stability."""
        self.game.lock_on_bonus_range = 45  # Increase lock on range for bosses
        self.game.los_lock_on_deactivate_time = 99  # Increase line of sight lock on deactivate time
        self.game.last_bonfire = self.env_args.bonfire
        self.game.player_stats = player_stats[self.ENV_ID]
        self.game.allow_moves = True
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_deaths = False
        self.game.allow_weapon_durability_dmg = False  # Weapons mustn't break during long sessions
        self.game.resume_game()

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
        if action in self.current_valid_actions():
            self._game_input.add_actions(actions[action])
        # We always call the update because it includes actions added by _lock_on for camera control
        # If no action was queued, the update is equivalent to a reset.
        self._game_input.update_input()

    def _step(self, action: int):
        """Perform the actual step ingame.

        Unpauses the game, takes 0.01s substeps ingame, checks if the step size is already reached,
        times animations, handles critical events, updates the internal state and resets the player
        and boss HP. Once the ``step_size`` length has been reached the game gets paused again and
        step postprocessing begins.

        Args:
            action: The action that is applied during this step.
        """
        self.game.game_speed = self._game_speed
        t_start = self.game.time
        previous_player_animation = self._internal_state.player_animation
        previous_boss_animation = self._internal_state.boss_animation
        boss_animation_start = t_start
        player_animation_start = t_start
        # Needs to be called AFTER resume game to apply roll/hits. Since roll and hit actions have
        # a blocking sleep call, we also begin the timing of animations before applying the action
        # so that this sleep is accounted for in the total step time.
        self._apply_action(action)
        # Offset of 0.01s to account for processing time of the loop
        while self.game.timed(self.game.time, t_start) < (max(self.step_size - 0.01, 1e-4)):
            boss_animation = self.game.get_boss_animation(self.ENV_ID)
            if boss_animation != previous_boss_animation:
                boss_animation_start = self.game.time
                previous_boss_animation = boss_animation
            player_animation = self.game.player_animation
            if player_animation != previous_player_animation:
                player_animation_start = self.game.time
                previous_player_animation = player_animation
            t_loop = self.game.time
            # Theoretically limits the loop to 1000 iterations / step. Effectively reduces the loop
            # to a few iterations as context switching allows the CPU to schedule other processes.
            # Disabled for now to increase loop timing precision
            # time.sleep(self.step_size / 1000.)
        self.game.pause_game()
        t_end = self.game.time
        game_state = self.game.get_state(self.ENV_ID, use_cache=True)
        # The animations might change between the last loop iteration and the game_state snapshot.
        # We therefore have to check one last time and update the animation durations accordingly
        if game_state.boss_animation != previous_boss_animation:
            boss_animation_start = t_loop
        if game_state.player_animation != previous_player_animation:
            player_animation_start = t_loop
        player_animation_td = self.game.timed(t_end, player_animation_start)
        boss_animation_td = self.game.timed(t_end, boss_animation_start)
        # During grab attacks, the lock cannot be established
        if not game_state.lock_on and game_state.player_animation not in ("ThrowAtk", "ThrowDef"):
            logger.debug("_step_check: Missing lock on detected")
            self._lock_on()
        else:
            self._lock_on_timer = 0
        if not self._step_check(game_state):
            self._handle_critical_game_state(game_state)
            return
        self._update_internal_game_state(game_state, player_animation_td, boss_animation_td)
        self._step_hook()

    def _step_hook(self):
        """Reset player and boss HP after a step has been made.

        Overwriting this hook in :class:`.SoulsEnvDemo` allows demos to skip the HP replenishment
        step for an unmodified episode.
        """
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
        coords = zip(self.env_args.coordinate_box_low, self.game.player_pose[:2],
                     self.env_args.coordinate_box_high)
        if not all(low < pos < high for low, pos, high in coords):
            logger.error("_step_check: Player outside of arena bounds")
            logger.error(game_state)
            raise InvalidPlayerStateError("Player outside of arena bounds")
        # Critical animations need special recovery routines
        if game_state.player_animation in critical_player_animations:
            return False
        # Fall detection by lower state space bound on z coordinate
        if self.env_args.coordinate_box_low[2] > game_state.player_pose[2]:
            return False
        # Unknown player animation. Shouldn't happen, add animation to tables!
        if game_state.player_animation not in player_animations:
            logger.warning(f"_step: Unknown player animation {game_state.player_animation}")
        if game_state.boss_animation not in boss_animations[self.ENV_ID]["all"]:
            logger.warning(f"_step: Unknown boss animation {game_state.boss_animation}")
        return True

    def _update_internal_game_state(self, game_state: GameState, player_animation_td: float,
                                    boss_animation_td: float):
        """Update the internal game state.

        Args:
            game_state: The current game state.
            player_animation_td: Player animation time difference for the animation duration update.
            boss_animation_td: Boss animation time difference for the animation duration update.

        Raises:
            ResetNeeded: Tried to update before resetting the environment first.
        """
        if self._internal_state is None or self.terminated:
            logger.error("_update_internal_game_state: SoulsEnv.step() called before reset")
            raise ResetNeeded("SoulsEnv.step() called before reset")
        # Save animation duration and HP
        if game_state.player_animation == self._internal_state.player_animation:
            player_animation_duration = self._internal_state.player_animation_duration
            player_animation_duration += player_animation_td
        else:
            player_animation_duration = player_animation_td
        if game_state.boss_animation == self._internal_state.boss_animation:
            boss_animation_duration = self._internal_state.boss_animation_duration
            boss_animation_duration += boss_animation_td
        else:
            boss_animation_duration = boss_animation_td
        player_hp, boss_hp = self._internal_state.player_hp, self._internal_state.boss_hp
        # Update animation count and HP
        self._internal_state = game_state
        self._internal_state.player_animation_duration = player_animation_duration
        self._internal_state.boss_animation_duration = boss_animation_duration
        self._internal_state.player_hp -= game_state.player_max_hp - player_hp
        self._internal_state.boss_hp -= game_state.boss_max_hp - boss_hp
        if self._internal_state.player_hp < 0:
            self._internal_state.player_hp = 0
        if self._internal_state.boss_hp < 0:
            self._internal_state.boss_hp = 0
        self.terminated = self._internal_state.player_hp == 0 or self._internal_state.boss_hp == 0

    def _handle_critical_game_state(self, game_state: GameState):
        """Handle critical game states.

        Args:
            game_state: The critical game state.
        """
        # Player is falling. Set player log HP to 0 and eagerly reset to prevent reload
        if game_state.player_pose[2] < self.env_args.coordinate_box_low[2]:
            game_state.player_hp = 0
            self._update_internal_game_state(game_state, self.step_size, self.step_size)
            self.game.reset_player_hp()
            self.game.reset_boss_hp(self.ENV_ID)
            self.game.player_pose = coordinates[self.ENV_ID]["player_init_pose"]
        if game_state.player_animation in critical_player_animations:
            game_state.player_hp = 0
            self._update_internal_game_state(game_state, self.step_size, self.step_size)

    def _lock_on(self, target_pose: np.ndarray | None = None):
        """Reestablish lock on by orienting the camera towards the boss and pressing lock on.

        If the optional target pose is given, the camera is instead oriented towards the coordinates
        of the target pose. If the lock on attempt fails the camera is still oriented towards the
        target pose. This way we imitate a lock on even if we can't actually establish lock on.

        Note:
            Lock on is disabled during the *ThrowAtk* and *ThrowDef* player animations because it is
            disabled on the game side for these animations.

        Warning:
            Lock on actions are only queued for execution on the subsequent environment step. Since
            the game is paused during :meth:`.SoulsEnv._lock_on`, we cannot restore the lock
            immediately as the camera does not move.

        Args:
            target_pose: The target pose towards which the camera should be oriented from its
                current position.
        """
        # During grab attacks, the lock cannot be established
        if self.game.player_animation not in ("ThrowAtk", "ThrowDef"):
            # Additional safeguard to make sure the player is currently not locked on
            if not self.game.lock_on:
                cpose = self.game.camera_pose
                if target_pose is None:
                    target_pose = self.game.get_boss_pose(self.ENV_ID) - self.game.player_pose
                normal = target_pose[:3] / np.linalg.norm(target_pose[:3])
                if np.dot(cpose[3:], normal) > 0.8 and self._lock_on_timer <= 0:
                    # Lock on is established on "button down", so we press once per 3 steps to make
                    # sure we don't get stuck after one bad press
                    self._game_input.add_action("lockon")
                    self._lock_on_timer = 3
                    return
                self._lock_on_timer -= 1
                dz = cpose[5] - normal[2]  # Camera pose is [x, y, z, nx, ny, nz], we need nz
                normal_angle = np.arctan2(*normal[:2])
                d_angle = wrap_to_pi(np.arctan2(cpose[3], cpose[4]) - normal_angle)
                if abs(dz) > 0.3:
                    self._game_input.add_action("cameradown" if dz > 0 else "cameraup")
                if abs(d_angle) > 0.3:
                    self._game_input.add_action("cameraleft" if d_angle > 0 else "cameraright")


class SoulsEnvDemo(SoulsEnv):
    """Demo class to show the performance of agents in the unaltered game.

    Demo envs do not reset the player and boss HP so that an episode resembles an actual boss fight.
    After a single episode, a hard reset of the game is necessary since either the player has died
    or the boss has been defeated.
    """

    def __init__(self, game_speed: int = 1.):
        """Initialize the demo environment.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
        """
        super().__init__(game_speed)

    def _update_internal_game_state(self, game_state: GameState, player_animation_td: float,
                                    boss_animation_td: float):
        if self._internal_state is None or self.terminated:
            logger.error("_update_internal_game_state: SoulsEnv.step() called before reset")
            raise ResetNeeded("SoulsEnv.step() called before reset")
        # Save animation duration and HP
        if game_state.player_animation == self._internal_state.player_animation:
            player_animation_duration = self._internal_state.player_animation_duration
            player_animation_duration += player_animation_td
        else:
            player_animation_duration = player_animation_td
        if game_state.boss_animation == self._internal_state.boss_animation:
            boss_animation_duration = self._internal_state.boss_animation_duration
            boss_animation_duration += boss_animation_td
        else:
            boss_animation_duration = boss_animation_td
        # Update animation count and HP
        self._internal_state = game_state
        self._internal_state.player_animation_duration = player_animation_duration
        self._internal_state.boss_animation_duration = boss_animation_duration
        if self._internal_state.player_hp < 0:
            self._internal_state.player_hp = 0
        if self._internal_state.boss_hp < 0:
            self._internal_state.boss_hp = 0
        self.terminated = self._internal_state.player_hp == 0 or self._internal_state.boss_hp == 0

    def _step_hook(self):
        """Omit HP replenishment steps to allow player and boss HP drop."""

    def _set_game_properties(self):
        """Enable player and boss deaths."""
        super()._set_game_properties()
        self.game.allow_deaths = True

    def _step(self, action):
        """Continue the game after finishing the demo."""
        super()._step(action)
        if self.terminated:
            self.game.resume_game()

    def _step_check(self, game_state: GameState) -> bool:
        """Check if game and player state are within expected values.

        Overwrite player death as non-critical.

        Args:
            game_state: The current game state.

        Returns:
            True if the check passed, else False.

        Raises:
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        if game_state.player_hp == 0:
            return True
        return super()._step_check(game_state)
