"""The SoulsGym environment for Margit, The Fell Omen.

The player and Margit always start from their respective start poses at full HP/SP/MP.

Note:
    The debug flags of Elden Ring are not mapped yet. In contrast to Dark Souls III, the
    environment cannot freeze the entities while it resets their poses.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

from soulsgym.envs.game_state import GameState
from soulsgym.envs.soulsenv import SoulsEnv
from soulsgym.envs.utils import max_retries
from soulsgym.exception import GameStateError, LockOnFailure, ResetError

if TYPE_CHECKING:
    from soulsgym.games import EldenRing

logger = logging.getLogger(__name__)


@dataclass
class MargitState(GameState):
    """Collect all game information for state tracking in a single data class.

    This class extends the base ``GameState`` with additional data members that are specific to the
    Margit fight.

    :meta private:
    """

    player_mp: int = 0
    player_max_mp: int = 0

    def copy(self) -> MargitState:
        """Create a copy of the state.

        Returns:
            A copy of itself.
        """
        return MargitState(**self.__dict__)


class MargitEnv(SoulsEnv):
    """Gymnasium environment class for the Margit bossfight.

    The environment uses the ground truth information from the game as observation space.
    """

    ENV_ID = "margit"
    BONFIRE = "Stormhill - Castleward Tunnel"
    # Broad arena bounds until the arena limits of Margit are calibrated
    ARENA_LIM_LOW = [-1e6, -1e6, -1e4, -3.1416]
    ARENA_LIM_HIGH = [1e6, 1e6, 1e4, 3.1416]
    CAM_SETUP_POSE = [0.0, 1.0, 0.0]
    MARGIT_MAX_HP = 4174
    HARD_RESET_INTERVAL = 900  # Reset the environment every 15 minutes
    POSE_REWRITE_INTERVAL = 0.5  # Seconds between repeated player pose writes during resets

    def __init__(
        self,
        game_speed: float = 1.0,
        phase: int = 1,
        random_player_pose: bool = False,
        skip_steps: bool = False,
        game_resolution: tuple[int, int] | None = None,
    ):
        """Initialize the observation and action spaces.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            phase: Set the boss phase. Only phase 1 of Margit is supported.
            random_player_pose: Flag to randomize the player pose on reset.
            skip_steps: Flag to skip steps while the player is disabled.
            game_resolution: Optional game-window resolution override.

        Raises:
            NotImplementedError: If ``phase`` is not 1.
        """
        if phase != 1:
            raise NotImplementedError("Only phase 1 of Margit is supported")
        # EldenRing needs to be open at this point
        super().__init__(
            game_speed=game_speed, skip_steps=skip_steps, game_resolution=game_resolution
        )
        # The state space consists of multiple spaces. These represent:
        # 1)       Boss phase. Episodes start in phase 1, only phase 1 is supported
        # 2 - 9)   Player and boss stats. In order: Player HP, player max HP, player SP, player max
        #          SP, player MP, player max MP, boss HP, boss max HP
        # 10 - 12) Player, boss and camera poses. In order: Player x, y, z, a, boss x, y, z, a,
        #          camera x, y, z, nx, ny, nz, where a represents the orientation and [nx ny nz]
        #          the camera plane normal
        # 13)      Player animation. -1 denotes unknown animations.
        # 14)      Player animation duration. We assume no animation takes longer than 10s
        # 15)      Boss animation. -1 denotes unknown animations.
        # 16)      Boss animation duration. We assume no animation takes longer than 10s.
        # 17)      Lock on flag. Either true or false.
        self.game: EldenRing  # Type hint only
        self._game_state: MargitState  # Type hint only
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
                "player_mp": spaces.Box(0, self.game.player_max_mp),
                "player_max_mp": spaces.Discrete(1, start=self.game.player_max_mp),
                "boss_hp": spaces.Box(0, self.MARGIT_MAX_HP),
                "boss_max_hp": spaces.Discrete(1, start=self.MARGIT_MAX_HP),
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
        self._player_died = False
        self._random_player_pose = random_player_pose
        self._last_hard_reset = time.time()

    @property
    def game_id(self) -> str:
        """Return the ID of the souls game that is required to run for this environment.

        Returns:
            The game ID.
        """
        return "EldenRing"

    @property
    def obs(self) -> dict[str, Any]:
        """Observation property of the environment.

        Returns:
            The current observation of the environment.
        """
        obs = self._game_state.as_dict()
        obs["player_hp"] = np.array([obs["player_hp"]], dtype=np.float32)
        obs["player_sp"] = np.array([obs["player_sp"]], dtype=np.float32)
        obs["player_mp"] = np.array([obs["player_mp"]], dtype=np.float32)
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

    def game_state(self) -> MargitState:
        """Read the current game state.

        Returns:
            The current game state.
        """
        game_state = MargitState(
            phase=self.phase,
            player_max_hp=self.game.player_max_hp,
            player_max_sp=self.game.player_max_sp,
            boss_max_hp=self.game.margit_max_hp,
        )
        game_state.lock_on = self.game.lock_on
        game_state.boss_pose = self.game.margit_pose
        game_state.boss_hp = self.game.margit_hp
        game_state.boss_animation = self.game.margit_animation
        game_state.boss_animation_duration = self.game.margit_animation_time
        game_state.player_animation = self.game.player_animation
        game_state.player_pose = self.game.player_pose
        game_state.camera_pose = self.game.camera_pose
        game_state.player_hp = self.game.player_hp
        game_state.player_sp = self.game.player_sp
        game_state.player_mp = self.game.player_mp
        game_state.player_max_mp = self.game.player_max_mp
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
        self._game_window.focus()
        if self._reload_required():
            self._reload()
        if self._arena_setup_required():
            self._arena_setup()
        self.game.time = 0  # Reset the total play time to avoid stuck timer on 999.99h
        self.game.game_speed = 3  # Increase game speed to speed up recovery
        self._entity_reset()
        self._camera_reset()
        self.game.pause()
        self.terminated = False
        self._game_state = self.game_state()
        return self.obs, self.info

    def _reload_required(self) -> bool:
        """Check if the environment needs to be reloaded.

        Returns:
            True if the environment needs to be reloaded, else False.
        """
        if self._player_died:  # The game is paused during the death, the reload respawns the player
            return True
        if not self.game.is_ingame:
            raise GameStateError("Player does not seem to be ingame")
        if self.game.margit_hp <= 0:
            return True
        if time.time() - self._last_hard_reset > self.HARD_RESET_INTERVAL:
            return True
        return False

    def _reload(self):
        """Set up the required flags and reload the game."""
        self._arena_init = False
        self.game.reload()
        self._player_died = False
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
        # Make sure to start around the bonfire. If the player is not idle, we also reload as a
        # precaution
        coords = self.game.data.coordinates[self.ENV_ID]
        d_pos = self.game.player_pose[:3] - coords["bonfire"][:3]
        idle = self.game.player_animation == self.game.PLAYER_IDLE_ANIMATION
        if np.linalg.norm(d_pos) > 10 or not idle:
            self.game.reload()
        self.game.player_pose = coords["fog_wall"]
        self.game.sleep(0.2)  # Wait for the teleport to finish
        d_pos = self.game.player_pose[:3] - coords["fog_wall"][:3]
        if np.linalg.norm(d_pos) > 2.0:
            raise ResetError("Player is not standing in front of the fog wall as expected")
        self._enter_fog_gate()
        self._arena_init = True
        self.game.pause()

    def _entity_reset(self):
        """Reset the player and boss HP and reset their poses."""
        self.game.reset_player_hp()
        self.game.reset_player_sp()
        self.game.reset_player_mp()
        self.game.reset_boss_hp("margit")
        coords = self.game.data.coordinates[self.ENV_ID]
        player_pose = coords["player_init_pose"]
        if self._random_player_pose:
            player_pose = random.choice(coords["player_init_poses_random"])
        self.game.player_pose = player_pose
        if self.game.lock_on:  # Release the lock on before resetting the poses
            self._game_input.single_action("lock_on", 0.005)
        t_start = t_pose = time.time()
        while not self._entity_reset_check(player_pose):
            # The pose setter moves the player relative to the current player position, which the
            # game updates one frame after a write. Repeating the write before the update moves the
            # player past the target, so we only repeat writes that the game has lost
            if time.time() - t_pose > self.POSE_REWRITE_INTERVAL:
                self.game.player_pose = player_pose
                t_pose = time.time()
            self.game.margit_pose = coords["boss_init_pose"]
            # If the poses can't be reset, we completely reset the environment by setting the
            # arena_init flag to False and raising a ResetError that implicitly starts the next
            # reset attempt through the @max_retries decorator
            if time.time() - t_start > 5:
                self._arena_init = False  # Make sure the arena is reset on the next reset call
                raise ResetError("Player or boss pose could not be reset")
            self.game.sleep(0.01)

    def _camera_reset(self):
        """Reset the camera pose and lock on to the boss."""
        self.game.game_speed = 3  # Speed up recovery
        t_start = time.time()
        while not self.game.lock_on:
            if time.time() - t_start > 3:
                raise LockOnFailure("Lock on to Margit could not be established")
            self._lock_on()
            self._game_input.update_input()
            self.game.sleep(0.01)
            self._game_input.reset()  # Prevent getting stuck if initial press is missed
            if not self._game_window.focused:
                self._game_window.focus()

    def _entity_reset_check(self, player_pose: np.ndarray) -> bool:
        """Check if the entity reset was successful.

        Args:
            player_pose: The targeted player start pose.

        Returns:
            True if the reset was successful, else False.
        """
        dist = np.linalg.norm(self.game.player_pose[:3] - player_pose[:3])
        if dist > 2:
            return False
        boss_init_pos = self.game.data.coordinates[self.ENV_ID]["boss_init_pose"][:3]
        if np.linalg.norm(self.game.margit_pose[:3] - boss_init_pos) > 3:
            return False
        if self.game.player_animation != self.game.PLAYER_IDLE_ANIMATION:
            return False
        if self.game.player_hp != self.game.player_max_hp:
            return False
        if self.game.player_sp != self.game.player_max_sp:
            return False
        if self.game.player_mp != self.game.player_max_mp:
            return False
        if self.game.margit_hp != self.game.margit_max_hp:
            return False
        return True

    def _enter_fog_gate(self):
        """Enter the fog gate by interacting with it and wait until the player is idle again."""
        coords = self.game.data.coordinates[self.ENV_ID]
        self._wait_until_idle(timeout=3, context="before entering the fog gate")
        self.game.camera_pose = self.CAM_SETUP_POSE
        self._game_input.single_action("interact")
        # The player turns towards the fog gate before the animation starts. We wait until the
        # player has passed the fog gate instead of waiting for a fixed time
        t_start = time.time()
        while np.linalg.norm(self.game.player_pose[:3] - coords["post_fog_wall"][:3]) > 2.0:
            if self.game.player_hp == 0:
                raise ResetError("Player has died while entering the fog gate")
            if time.time() - t_start > 5:
                raise ResetError("Player has not entered the fog wall as expected")
            self.game.sleep(0.1)
            self.game.clear_cache()
        self._wait_until_idle(timeout=10, context="after entering the fog gate")

    def _wait_until_idle(self, timeout: float, context: str):
        """Wait until the player is idle and reacts to inputs again.

        The game ignores inputs during animations, e.g. while the player falls after a teleport.

        Args:
            timeout: The maximum waiting time in seconds.
            context: The reset step for the error messages.

        Raises:
            ResetError: The player has died or is not idle within the timeout.
        """
        t_start = time.time()
        while True:
            if self.game.player_hp == 0:
                raise ResetError(f"Player has died {context}")
            animation = self.game.player_animation
            if animation == self.game.PLAYER_IDLE_ANIMATION:
                return
            if time.time() - t_start > timeout:
                raise ResetError(
                    f"Player is not idle {context}, current animation ID is {animation}"
                )
            self.game.sleep(0.1)
            # Clear the cache in case the player has died to prevent reading stale addresses
            self.game.clear_cache()

    def _set_game_properties(self):
        """Set general game properties that help gym stability.

        The debug flags of Elden Ring are not mapped yet. In contrast to Dark Souls III, the lock on
        range, the flags for attacks, hits, moves and deaths, and the weapon durability are
        therefore left untouched.
        """
        self.game.last_bonfire = self.BONFIRE
        self.game.player_stats = self.game.data.player_stats[self.ENV_ID]
        self.game.resume()

    def _step_hook(self):
        """Reset the player and boss HP and the player MP after a step has been made."""
        super()._step_hook()
        self.game.reset_player_mp()

    def _step_check(self, game_state: GameState) -> bool:
        """Check if game and player state are within expected values.

        The debug flags of Elden Ring are not mapped, so the environment cannot prevent deaths, e.g.
        from a fall out of the arena. In contrast to Dark Souls III, a player death is therefore a
        critical game state instead of an error.

        Args:
            game_state: The current game state.

        Returns:
            True if the check passed, else False.
        """
        if game_state.player_hp == 0:
            logger.info("_step_check: Player has died in the game")
            return False
        return super()._step_check(game_state)

    def _handle_critical_game_state(self, game_state: GameState):
        """Handle critical game states.

        A player death ends the episode. The next reset reloads the game to respawn the player.

        Args:
            game_state: The critical game state.
        """
        if game_state.player_hp == 0:
            self._player_died = True
            self._update_game_state(game_state, self.step_size, self.step_size)
            return
        super()._handle_critical_game_state(game_state)

    def close(self):
        """Unpause the game and reload.

        Note:
            Does not wait for reload to complete.
        """
        self.game.resume()
        self._game_input.reset()
        self.game.player_hp = 0  # Kill player to force game reload. Don't wait for completion

    @staticmethod
    def compute_reward(game_state: MargitState, next_game_state: MargitState) -> float:
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
            base_reward = 0.0
        return boss_reward + player_reward + base_reward
