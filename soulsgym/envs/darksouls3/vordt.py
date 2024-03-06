"""The SoulsGym environment for Vordt of the Boreal Valley.

The player and Vordt always start from their respective start poses at full HP/SP. The player starts
with the stats and weapons as configured in <TODO: add config>. We do not allow shield blocking or
two handing at this point, although this can easily be supported. Parrying is enabled.

Note:
    Phase 2 of the boss fight is available by setting the environment keyword argument ``phase``.
    See :mod:`~.envs` for details.
"""
from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from gymnasium import spaces
from soulsgym.envs.game_state import GameState

from soulsgym.envs.soulsenv import SoulsEnv

if TYPE_CHECKING:
    from soulsgym.games import DarkSoulsIII

logger = logging.getLogger(__name__)


class VordtState(GameState):
    """Collect all game information for state tracking in a single data class.

    This class extends the base ``GameState`` with additional data members that are specific to the
    Iudex Gundyr fight.
    """


class VordtEnv(SoulsEnv):
    """The SoulsGym environment for Vordt of the Boreal Valley."""

    ENV_ID = "vordt"
    BONFIRE = "Dancer of the Boreal Valley"
    ARENA_LIM_LOW = [13., -7., -27., -3.1416]
    ARENA_LIM_HIGH = [42., 46., -15., 3.1416]
    CAM_SETUP_POSE = [0., -1., 0.]
    VORDT_MAX_HP = 1328

    def __init__(self, game_speed: float = 1., phase: int = 1):
        """Initialize the observation and action spaces.

        Args:
            game_speed: The speed of the game during :meth:`.SoulsEnv.step`. Defaults to 1.0.
            phase: The phase of the boss fight. Either 1 or 2 for Vordt. Defaults to 1.
        """
        super().__init__(game_speed=game_speed)
        self.game: DarkSoulsIII  # Type hint only
        self.phase = phase
        pose_box_low = np.array(self.ARENA_LIM_LOW, dtype=np.float32)
        pose_box_high = np.array(self.ARENA_LIM_HIGH, dtype=np.float32)
        cam_box_low = np.array(self.ARENA_LIM_LOW[:3] + [-1, -1, -1], dtype=np.float32)
        cam_box_high = np.array(self.ARENA_LIM_HIGH[:3] + [1, 1, 1], dtype=np.float32)
        player_animations = self.game.data.player_animations
        boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(2, start=1),
            "player_hp": spaces.Box(0, self.game.player_max_hp),
            "player_max_hp": spaces.Discrete(1, start=self.game.player_max_hp),
            "player_sp": spaces.Box(0, self.game.player_max_sp),
            "player_max_sp": spaces.Discrete(1, start=self.game.player_max_sp),
            "boss_hp": spaces.Box(0, self.VORDT_MAX_HP),
            "boss_max_hp": spaces.Discrete(1, start=self.VORDT_MAX_HP),
            "player_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
            "boss_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
            "camera_pose": spaces.Box(cam_box_low, cam_box_high, dtype=np.float32),
            "player_animation": spaces.Discrete(len(player_animations) + 1, start=-1),
            "player_animation_duration": spaces.Box(0., 10.),
            "boss_animation": spaces.Discrete(len(boss_animations) + 1, start=-1),
            "boss_animation_duration": spaces.Box(0., 10.),
            "lock_on": spaces.Discrete(2),
            "frost_resistance": spaces.Box(0., 1.),
            "frost_effect": spaces.Box(0., 1.),
        })
        self.action_space = spaces.Discrete(len(self.game.data.actions))
        assert phase in (1, 2)
        self.phase = phase
        self._arena_init = False
        self._phase_init = False
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
    def obs(self) -> dict:
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
        obs["frost_resistance"] = np.array([self.game.player_frost_resistance], np.float32)
        obs["frost_effect"] = np.array([self.game.player_frost_effect], np.float32)
        return obs

    @property
    def info(self) -> dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        return {"allowed_actions": self.current_valid_actions()}

    def game_state(self) -> VordtState:
        """Read the current game state.

        Returns:
            The current game state.
        """
        game_state = VordtState(player_max_hp=self.game.player_max_hp,
                                player_max_sp=self.game.player_max_sp,
                                boss_max_hp=self.game.vordt_max_hp)
        game_state.lock_on = self.game.lock_on
        game_state.boss_pose = self.game.vordt_pose
        game_state.boss_hp = self.game.vordt_hp
        game_state.boss_animation = self.game.vordt_animation
        game_state.player_animation = self.game.player_animation
        game_state.player_pose = self.game.player_pose
        game_state.camera_pose = self.game.camera_pose
        game_state.player_hp = self.game.player_hp
        game_state.player_sp = self.game.player_sp
        return game_state.copy()

    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[dict, dict]:
        """Reset the environment to its initial state.

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
        self._entity_reset()
        self._camera_reset()
        self.game.pause()
        self.terminated = False
        self._game_state = self.game_state()
        return self.obs, self.info

    def _reload_required(self) -> bool:
        """Check if a game reload is required.

        Returns:
            True if a game reload is required, False otherwise.
        """
        if not self.game.vordt_flags:  # Boss is dead, not encountered etc.
            return True
        if time.time() - 900 > self._last_hard_reset:  # Reset every 15 minutes
            return True
        if not self._arena_init:  # Player is not already in the arena and not at the bonfire
            bonfire_pos = self.game.data.coordinates[self.ENV_ID]["bonfire"][:3]
            if np.linalg.norm(self.game.player_pose[:3] - bonfire_pos) > 10:
                return True
        return False

    def _reload(self):
        """Reload the environment completely."""
        self.game.vordt_flags = True
        self.game.game_speed = 3  # Faster death
        self.game.reload()
        self._last_hard_reset = time.time()
        self._arena_init = False
        self._phase_init = False
        self.game.game_speed = self._game_speed

    def _arena_setup_required(self) -> bool:
        """Check if the arena setup is required.

        Returns:
            True if the arena setup is required, False otherwise.
        """
        return not self._arena_init

    def _arena_setup(self):
        """Teleport the player to the arena, enter the fog gate and warp to the start position."""
        self.game.game_speed = 3  # Faster teleport
        self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["fog_wall"]
        self.game.sleep(0.2)  # Wait for the teleport to finish
        while not self.game.player_animation == "Idle":
            self.game.sleep(0.1)  # Sometimes the teleport triggers a fall animation
        # Enter the fog gate
        self.game.camera_pose = self.CAM_SETUP_POSE
        self._game_input.single_action("interact")
        self.game.sleep(0.2)  # Wait for the fog gate animation to start
        while not self.game.player_animation == "Idle":
            self.game.sleep(0.1)
        self.game.game_speed = self._game_speed
        self._arena_init = True

    def _phase_setup_required(self) -> bool:
        """Check if the phase setup is required.

        Returns:
            True if the phase setup is required, False otherwise.
        """
        return self._phase_init

    def _phase_setup(self):
        """Trigger the phase transition."""
        if self.phase == 2:
            self.game.vordt_hp = 100
            self.game.allow_attacks = True
            self.game.game_speed = 3
        while not self.game.vordt_animation == "Idle":
            self.game.sleep(0.1)
        while self.game.vordt_animation == "Idle":
            self.game.sleep(0.1)
        self.game.allow_attacks = False
        self.game.pause()
        self.game.reset_boss_hp("vordt")
        self._phase_init = True

    def _entity_reset(self):
        """Reset the player and boss entities."""
        self.game.allow_attacks = False
        self.game.allow_moves = False
        self.game.game_speed = 3  # Faster reset
        self.game.player_frost_resistance = 1.
        # self.game.player_frost_effect = 0.  TODO: Can't be set to 0, maybe change address?
        while not self._entity_reset_complete():
            self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["player_init_pose"]
            self.game.vordt_pose = self.game.data.coordinates[self.ENV_ID]["boss_init_pose"]
            self.game.sleep(0.01)
        self.game.pause()
        self.game.allow_attacks = True
        self.game.allow_moves = True

    def _entity_reset_complete(self) -> bool:
        """Check if the player and boss have been successfully reset.

        Returns:
            True if the entities have been reset, False otherwise.
        """
        if not self.game.player_animation == "Idle":
            return False
        if self.game.vordt_animation not in ("Attack9910", "IdleBattle"):
            return False
        desired_pose = self.game.data.coordinates[self.ENV_ID]["player_init_pose"]
        if np.linalg.norm(self.game.player_pose - desired_pose) > 0.5:
            return False
        desired_pose = self.game.data.coordinates[self.ENV_ID]["boss_init_pose"]
        if np.linalg.norm(self.game.vordt_pose - desired_pose) > 0.5:
            return False
        return True

    def _camera_reset(self):
        """Reset the camera to a locked on state."""
        self.game.resume()
        while not self.game.lock_on:
            self._lock_on()
            self._game_input.update_input()
            self.game.sleep(0.01)
            self._game_input.reset()  # Prevent getting stuck if initial press is missed
            if not self._game_window.focused:
                self._game_window.focus_application()

    @staticmethod
    def compute_reward(game_state: VordtState, next_game_state: VordtState) -> float:
        """Compute the reward from the current game state and the next game state.

        Args:
            game_state: The game state before the step.
            next_game_state: The game state after the step.

        Returns:
            The reward for the provided game states.
        """
        boss_reward = (game_state.boss_hp - next_game_state.boss_hp) / game_state.boss_max_hp
        player_hp_diff = (next_game_state.player_hp - game_state.player_hp)
        player_reward = player_hp_diff / game_state.player_max_hp
        base_reward = 0.
        if next_game_state.boss_hp == 0 or next_game_state.player_hp == 0:
            base_reward = 0.1 if next_game_state.boss_hp == 0 else -0.1
        return boss_reward + player_reward + base_reward
