"""The SoulsGym environment for Iudex Gundyr.

The player and Iudex always start from their respective start poses at full HP/SP. A fall from
the cliff is considered an instantaneous death and is eagerly reset to avoid falling out of the
world. The player starts with the knight base stats and the default starting weapons without any
upgrades. We do not allow shield blocking or two handing at this point, although this can easily
be supported. Parrying is enabled.

Note:
    Phase 2 of the boss fight is available by setting the environment keyword argument ``phase``.
    See :mod:`~.envs` for details.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

from soulsgym.envs.game_state import GameState
from soulsgym.envs.soulsenv import SoulsEnv, SoulsEnvDemo
from soulsgym.envs.utils import max_retries
from soulsgym.exception import GameStateError, ResetError

if TYPE_CHECKING:
    from soulsgym.games import DarkSoulsIII

logger = logging.getLogger(__name__)


class IudexState(GameState):
    """Collect all game information for state tracking in a single data class.

    This class extends the base ``GameState`` with additional data members that are specific to the
    Iudex Gundyr fight.

    :meta private:
    """


class IudexEnv(SoulsEnv):
    """Gymnasium environment class for the Iudex Gundyr bossfight.

    The environment uses the ground truth information from the game as observation space.
    """

    ENV_ID = "iudex"
    BONFIRE = "Cemetery of Ash"
    ARENA_LIM_LOW = [110.0, 540.0, -73.0, -3.1416]
    ARENA_LIM_HIGH = [190.0, 640.0, -55.0, 3.1416]
    CAM_SETUP_POSE = [0.378, 0.926, 0.0]
    IUDEX_MAX_HP = 1037  # When we are not in Cemetery of Ash, the boss HP read can be incorrect
    HARD_RESET_INTERVAL = 900  # Reset the environment every 15 minutes

    def __init__(
        self,
        game_speed: float = 1.0,
        phase: int = 1,
        random_player_pose: bool = False,
        skip_steps: bool = False,
    ):
        """Initialize the observation and action spaces.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            phase: Set the boss phase. Either 1 or 2 for Iudex.
            random_player_pose: Flag to randomize the player pose on reset.
            skip_steps: Flag to skip steps while the player is disabled.
        """
        assert phase in (1, 2)
        # DarkSoulsIII needs to be open at this point
        super().__init__(game_speed=game_speed, skip_steps=skip_steps)
        # The state space consists of multiple spaces. These represent:
        # 1)      Boss phase. Either 1 or 2 for Iudex
        # 2 - 7)  Player and boss stats. In order: Player HP, player max HP, player SP, player max
        #         SP, boss HP, boss max HP
        # 8 - 10) Player, boss and camera poses. In order: Player x, y, z, a, boss x, y, z, a,
        #         camera x, y, z, nx, ny, nz, where a represents the orientation and [nx ny nz]
        #         the camera plane normal
        # 11)     Player animation. -1 denotes unknown or critical animations.
        # 12)     Player animation duration. We assume no animation takes longer than 10s
        # 13)     Boss animation. -1 denotes unknown or critical animations.
        # 14)     Boss animation duration. We assume no animation takes longer than 10s.
        # 15)     Lock on flag. Either true or false.
        self.game: DarkSoulsIII  # Type hint only
        self._game_state: IudexState  # Type hint only
        pose_box_low = np.array(self.ARENA_LIM_LOW, dtype=np.float32)
        pose_box_high = np.array(self.ARENA_LIM_HIGH, dtype=np.float32)
        cam_box_low = np.array(self.ARENA_LIM_LOW[:3] + [-1, -1, -1], dtype=np.float32)
        cam_box_high = np.array(self.ARENA_LIM_HIGH[:3] + [1, 1, 1], dtype=np.float32)
        player_animations = self.game.data.player_animations
        boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        self.observation_space = spaces.Dict(
            {
                "phase": spaces.Discrete(2, start=1),
                "player_hp": spaces.Box(0, self.game.player_max_hp),
                "player_max_hp": spaces.Discrete(1, start=self.game.player_max_hp),
                "player_sp": spaces.Box(0, self.game.player_max_sp),
                "player_max_sp": spaces.Discrete(1, start=self.game.player_max_sp),
                "boss_hp": spaces.Box(0, self.IUDEX_MAX_HP),
                "boss_max_hp": spaces.Discrete(1, start=self.IUDEX_MAX_HP),
                "player_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
                "boss_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
                "camera_pose": spaces.Box(cam_box_low, cam_box_high, dtype=np.float32),
                "player_animation": spaces.Discrete(len(player_animations) + 1, start=-1),
                "player_animation_duration": spaces.Box(0.0, 10.0),
                "boss_animation": spaces.Discrete(len(boss_animations) + 1, start=-1),
                "boss_animation_duration": spaces.Box(0.0, 10.0),
                "lock_on": spaces.Discrete(2),
            }
        )
        self.action_space = spaces.Discrete(len(self.game.data.actions))
        self.phase = phase
        self._arena_init = False
        self._phase_init = False
        self._random_player_pose = random_player_pose
        # We keep track of the last time the environment has been reset completely. After several
        # hours of gameplay, the game input starts to lag and the agent's actions are not executed
        # properly anymore. We therefore reset the environment every 15 minutes to avoid an
        # unintended performance degradation
        self._last_hard_reset = time.time()

    @property
    def game_id(self) -> str:
        """Return the ID of the souls game that is required to run for this environment.

        Returns:
            The game ID.
        """
        return "DarkSoulsIII"

    @property
    def obs(self) -> dict[str, Any]:
        """Observation property of the environment.

        Returns:
            The current observation of the environment.
        """
        obs = self._game_state.as_dict()
        obs["player_hp"] = np.array([obs["player_hp"]], dtype=np.float32)
        obs["player_sp"] = np.array([obs["player_sp"]], dtype=np.float32)
        obs["boss_hp"] = np.array([obs["boss_hp"]], dtype=np.float32)
        # Default animation ID for unknown animations is -1
        _player_animations = self.game.data.player_animations
        obs["player_animation"] = _player_animations.get(obs["player_animation"], {"ID": -1})["ID"]
        _boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        obs["boss_animation"] = _boss_animations.get(obs["boss_animation"], {"ID": -1})["ID"]
        obs["player_animation_duration"] = np.array([obs["player_animation_duration"]], np.float32)
        obs["boss_animation_duration"] = np.array([obs["boss_animation_duration"]], np.float32)
        obs["player_pose"] = obs["player_pose"].astype(np.float32)
        obs["boss_pose"] = obs["boss_pose"].astype(np.float32)
        obs["camera_pose"] = obs["camera_pose"].astype(np.float32)
        return obs

    @property
    def info(self) -> dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        action_mask[self.current_valid_actions()] = True
        return {"action_mask": action_mask}

    def game_state(self) -> IudexState:
        """Read the current game state.

        Returns:
            The current game state.
        """
        game_state = IudexState(
            player_max_hp=self.game.player_max_hp,
            player_max_sp=self.game.player_max_sp,
            boss_max_hp=self.game.iudex_max_hp,
        )
        game_state.lock_on = self.game.lock_on
        game_state.boss_pose = self.game.iudex_pose
        game_state.boss_hp = self.game.iudex_hp
        game_state.boss_animation = self.game.iudex_animation
        game_state.player_animation = self.game.player_animation
        game_state.player_pose = self.game.player_pose
        game_state.camera_pose = self.game.camera_pose
        game_state.player_hp = self.game.player_hp
        game_state.player_sp = self.game.player_sp
        return game_state.copy()

    @max_retries(retries=3)
    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[dict, dict]:
        """Reset the environment to the beginning of an episode.

        Args:
            seed: Random seed. Required by gymnasium, but does not apply to SoulsGyms.
            options: Options argument required by gymnasium. Not used in SoulsGym.

        Returns:
            A tuple of the first game state and the info dict after the reset.
        """
        self._game_input.reset()
        self._game_window.focus_application()
        if self._reload_required():
            self._reload()
        if self._arena_setup_required():
            self._arena_setup()
        if self._phase_setup_required():
            self._phase_setup()
        self.game.allow_attacks = False
        self.game.allow_hits = False
        self.game.allow_moves = False
        self.game.time = 0  # Reset the total play time to avoid stuck timer on 999.99h
        self.game.game_speed = 3  # Increase game speed to speed up recovery
        self._entity_reset()
        self._camera_reset()
        self.game.pause()
        self.terminated = False
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_moves = True
        self._game_state = self.game_state()
        return self.obs, self.info

    def _reload_required(self) -> bool:
        """Check if the environment needs to be reloaded.

        Returns:
            True if the environment needs to be reloaded, else False.
        """
        if not self.game.is_ingame:
            raise GameStateError("Player does not seem to be ingame")
        if not self.game.iudex_flags:
            return True
        if time.time() - self._last_hard_reset > self.HARD_RESET_INTERVAL:
            return True
        return False

    def _reload(self):
        """Set up the required flags and reload the game."""
        self.game.iudex_flags = True
        self._arena_init = False
        self._phase_init = False
        self.game.reload()
        self._last_hard_reset = time.time()

    def _arena_setup_required(self) -> bool:
        """Check if the arena needs to be set up.

        Returns:
            True if the arena needs to be set up, else False.
        """
        return not self._arena_init

    @max_retries(retries=5)
    def _arena_setup(self):
        """Set up the arena."""
        self.game.game_speed = 3  # Increase game speed to speed up player actions
        # Make sure to start around the bonfire. In case the player has entered the arena on a
        # previous try, Iudex has to deaggro before the player can enter the arena again. This
        # forces us to reload. If the player is not idle, we also reload as a precaution
        d_pos = self.game.player_pose[:3] - self.game.data.coordinates[self.ENV_ID]["bonfire"][:3]
        if np.linalg.norm(d_pos) > 10 or self.game.player_animation != "Idle":
            self.game.reload()
        self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["fog_wall"]
        self.game.sleep(0.2)  # Wait for the teleport to finish and the camera to settle
        d_pos = self.game.player_pose[:3] - self.game.data.coordinates[self.ENV_ID]["fog_wall"][:3]
        if np.linalg.norm(d_pos) > 0.1:
            raise ResetError("Player is not standing in front of the fog wall as expected")
        self._enter_fog_gate()
        self._arena_init = True
        self._phase_init = False  # When we reload, the phase is automatically reset to 1
        self.game.pause()

    def _phase_setup_required(self) -> bool:
        """Check if the boss phase needs to be set up.

        Returns:
            True if the boss phase needs to be set up, else False.
        """
        return self.phase == 2 and not self._phase_init

    def _phase_setup(self):
        """Reduce Iudex HP to trigger phase transition and wait for completion."""
        self.game.iudex_hp = 100
        self.game.allow_attacks = True  # Iudex needs to attack for the transition
        self.game.game_speed = 3
        while not self.game.iudex_animation == "Attack1500":  # Wait for transition animation
            self.game.sleep(0.1)
        while self.game.iudex_animation == "Attack1500":  # Wait for transition animation to end
            self.game.sleep(0.1)
        self.game.allow_attacks = False
        self.game.pause()
        self.game.reset_boss_hp("iudex")
        self._phase_init = True

    def _entity_reset(self):
        """Reset the player and boss HP and reset their poses."""
        self.game.reset_player_hp()
        self.game.reset_player_sp()
        self.game.reset_boss_hp("iudex")
        player_pose = self.game.data.coordinates[self.ENV_ID]["player_init_pose"]
        if self._random_player_pose:  # Sample player pose uniformly at random around Iudex
            init_poses = self.game.data.coordinates[self.ENV_ID]["player_init_poses_random"]
            player_pose = random.choice(init_poses)
        self.game.player_pose = player_pose
        # When the arena is reset in between episodes, the player is possibly locked on to Iudex or
        # a unit outside the arena. This changes the player's orientation and prevents the
        # successful teleport to the initial pose. We therefore have to release lock on
        if self.game.lock_on:
            self._game_input.single_action("lock_on", 0.005)
        tstart = time.time()
        while not self._entity_reset_check(player_pose):
            self.game.player_pose = player_pose
            self.game.iudex_pose = self.game.data.coordinates[self.ENV_ID]["boss_init_pose"]
            # On rare occasions, the player can still get stuck with lock ons outside the arena, or
            # races lead to unexpected bugs. In that case, we completely reset the environment by
            # setting the arena_init flag to False and raising a ResetError that implicitly starts
            # the next reset attempt through the @max_retries decorator
            if time.time() - tstart > 5:
                self._arena_init = False  # Make sure the arena is reset on the next reset call
                raise ResetError("Player or boss pose could not be reset")
            self.game.sleep(0.01)

    def _camera_reset(self):
        """Reset the camera pose and lock on to the boss."""
        self.game.game_speed = 3  # Speed up recovery
        while not self.game.lock_on:
            self._lock_on()
            self._game_input.update_input()
            self.game.sleep(0.01)
            self._game_input.reset()  # Prevent getting stuck if initial press is missed
            if not self._game_window.focused:
                self._game_window.focus_application()

    def _entity_reset_check(self, player_pose: np.ndarray) -> bool:
        """Check if the entity reset was successful.

        Args:
            player_pose: The targeted player start pose.

        Returns:
            True if the reset was successful, else False.
        """
        dist = np.linalg.norm(self.game.player_pose[:3] - player_pose[:3])
        if dist > 1:
            return False
        boss_init_pos = self.game.data.coordinates[self.ENV_ID]["boss_init_pose"][:3]
        if np.linalg.norm(self.game.iudex_pose[:3] - boss_init_pos) > 2:
            return False
        boss_animation = self.game.iudex_animation
        if not any([a in boss_animation for a in ("Walk", "Idle")]) and boss_animation != "":
            return False
        if self.game.player_animation != "Idle":
            return False
        if self.game.player_hp != self.game.player_max_hp:
            return False
        if self.game.iudex_hp != self.game.iudex_max_hp:
            return False
        return True

    def _enter_fog_gate(self):
        """Enter the fog gate."""
        self.game.camera_pose = self.CAM_SETUP_POSE
        self._game_input.single_action("interact")
        while not self.game.player_animation == "Idle":
            self.game.sleep(0.1)
            # Clear the cache in case the player has died to prevent reading stale addresses
            self.game.clear_cache()
        post_fog_wall_pos = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"][:3]
        if np.linalg.norm(self.game.player_pose[:3] - post_fog_wall_pos) > 0.2:
            raise ResetError("Player has not entered the fog wall as expected")
        if self.game.player_hp == 0:
            raise ResetError("Player has died during reset")

    @staticmethod
    def compute_reward(game_state: IudexState, next_game_state: IudexState) -> float:
        """Compute the reward from the current game state and the next game state.

        Args:
            game_state: The game state before the step.
            next_game_state: The game state after the step.

        Returns:
            The reward for the provided game states.
        """
        boss_reward = (game_state.boss_hp - next_game_state.boss_hp) / game_state.boss_max_hp
        player_hp_diff = next_game_state.player_hp - game_state.player_hp
        player_reward = player_hp_diff / game_state.player_max_hp
        if next_game_state.boss_hp == 0 or next_game_state.player_hp == 0:
            base_reward = 0.1 if next_game_state.boss_hp == 0 else -0.1
        else:
            # Experimental: Reward for moving towards the arena center, no reward within 4m distance
            d_center_now = np.linalg.norm(
                next_game_state.player_pose[:2] - np.array([139.0, 596.0])
            )
            d_center_prev = np.linalg.norm(game_state.player_pose[:2] - np.array([139.0, 596.0]))
            base_reward = 0.01 * (d_center_prev - d_center_now) * (d_center_now > 4)
        return boss_reward + player_reward + base_reward


class IudexImgEnv(IudexEnv):
    """Gymnasium environment class for the Iudex Gundyr bossfight.

    The environment uses the ground truth information from the game as observation space. The boss
    HP are available in the info dict.
    """

    def __init__(
        self,
        game_speed: int = 1.0,
        phase: int = 1,
        random_player_pose: bool = False,
        skip_steps: bool = False,
        resolution: tuple[int, int] = (90, 160),
    ):
        """Overwrite the observation space to use the game image.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            phase: Set the boss phase. Either 1 or 2 for Iudex.
            random_player_pose: Flag to randomize the player pose on reset.
            skip_steps: Flag to skip steps while the player is disabled.
            resolution: The resolution of the game image.
        """
        super().__init__(game_speed, phase, random_player_pose, skip_steps=skip_steps)
        assert len(resolution) == 2
        self.observation_space = spaces.Box(
            low=0, high=255, shape=resolution + (3,), dtype=np.uint8
        )
        self.game.img_resolution = resolution

    @property
    def obs(self) -> np.ndarray:
        """Return the current observation."""
        return self.game.img

    @property
    def info(self) -> dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        action_mask[self.current_valid_actions()] = True
        return {"action_mask": action_mask, "boss_hp": self._game_state.boss_hp}


class IudexEnvDemo(SoulsEnvDemo, IudexEnv):
    """Demo environment for the Iudex Gundyr fight.

    Covers both phases. Player and boss loose HP, and the episode does not reset.
    """

    def __init__(self, game_speed: float = 1.0, random_player_pose: bool = False):
        """Initialize the demo environment.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            random_player_pose: Flag to randomize the player pose on reset.
        """
        super().__init__(game_speed)
        # IudexEnv can't be called with all arguments, so we have to set it manually after __init__
        self._random_player_pose = random_player_pose
        self.phase = 1

    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[dict, dict]:
        """Reset the environment to the beginning of an episode.

        Args:
            seed: Random seed. Required by gymnasium, but does not apply to SoulsGyms.
            options: Options argument required by gymnasium. Not used in SoulsGym.

        Returns:
            A tuple of the first game state and the info dict after the reset.
        """
        self._game_input.reset()
        self.game.reload()
        self._is_init = False
        self.phase = 1
        return super().reset()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """Perform a step forward in the environment with a given action.

        Each step advances the ingame time by `step_size` seconds. The game is paused before and
        after the step.

        Args:
            action: The action that is applied during this step.

        Returns:
            A tuple of the next game state, the reward, the terminated flag, the truncated flag, and
            an additional info dictionary.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        if self._game_state.boss_animation == "Attack1500":  # Phase change animation
            self.phase = 2
        obs["phase"] = self.phase
        return obs, reward, terminated, truncated, info
