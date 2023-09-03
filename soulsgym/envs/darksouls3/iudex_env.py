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
import logging
import random
import time
from typing import Any, Tuple, Dict

from pymem.exception import MemoryReadError
import numpy as np
from gymnasium import spaces
from gymnasium.error import RetriesExceededError

from soulsgym.envs.soulsenv import SoulsEnv, SoulsEnvDemo
from soulsgym.core.game_state import GameState
from soulsgym.exception import GameStateError

logger = logging.getLogger(__name__)


class IudexEnv(SoulsEnv):
    """``IudexEnv`` implements the environment of the Iudex Gundyr boss fight."""

    ENV_ID = "iudex"

    def __init__(self, game_speed: int = 1., phase: int = 1, init_pose_randomization: bool = False):
        """Define the state space.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            phase: Set the boss phase. Either 1 or 2 for Iudex.
            init_pose_randomization: Flag to randomize the player pose on reset.
        """
        # DarkSoulsIII needs to be open at this point
        super().__init__(game_speed=game_speed)
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
        pose_box_low = np.array(self.env_args.coordinate_box_low, dtype=np.float32)
        pose_box_high = np.array(self.env_args.coordinate_box_high, dtype=np.float32)
        cam_box_low = np.array(self.env_args.coordinate_box_low[:3] + [-1, -1, -1],
                               dtype=np.float32)
        cam_box_high = np.array(self.env_args.coordinate_box_high[:3] + [1, 1, 1], dtype=np.float32)
        player_animations = self.game.data.player_animations
        boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(2, start=1),
            "player_hp": spaces.Box(0, self.env_args.player_max_hp),
            "player_max_hp": spaces.Discrete(1, start=self.env_args.player_max_hp),
            "player_sp": spaces.Box(0, self.env_args.player_max_sp),
            "player_max_sp": spaces.Discrete(1, start=self.env_args.player_max_sp),
            "boss_hp": spaces.Box(0, self.env_args.boss_max_hp),
            "boss_max_hp": spaces.Discrete(1, start=self.env_args.boss_max_hp),
            "player_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
            "boss_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
            "camera_pose": spaces.Box(cam_box_low, cam_box_high, dtype=np.float32),
            "player_animation": spaces.Discrete(len(player_animations) + 1, start=-1),
            "player_animation_duration": spaces.Box(0., 10.),
            "boss_animation": spaces.Discrete(len(boss_animations) + 1, start=-1),
            "boss_animation_duration": spaces.Box(0., 10.),
            "lock_on": spaces.Discrete(2)
        })
        self.action_space = spaces.Discrete(len(self.game.data.actions))
        assert phase in (1, 2)
        self.phase = phase
        self._phase_init = False
        self._init_pose_randomization = init_pose_randomization

    @property
    def game_id(self):
        return "DarkSoulsIII"

    @property
    def obs(self) -> Dict:
        """Observation property of the environment.

        Returns:
            The current observation of the environment.
        """
        obs = self._internal_state.as_dict()
        obs["player_hp"] = np.array([obs["player_hp"]], dtype=np.float32)
        obs["player_sp"] = np.array([obs["player_sp"]], dtype=np.float32)
        obs["boss_hp"] = np.array([obs["boss_hp"]], dtype=np.float32)
        # Default animation ID for unknown animations is -1
        _player_animations = self.game.data.player_animations
        obs["player_animation"] = _player_animations.get(obs["player_animation"], {"ID": -1})["ID"]
        _boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        obs["boss_animation"] = _boss_animations.get(obs["boss_animation"], {"ID": -1})["ID"]
        obs["player_animation_duration"] = np.array([obs["player_animation_duration"]],
                                                    dtype=np.float32)
        obs["boss_animation_duration"] = np.array([obs["boss_animation_duration"]],
                                                  dtype=np.float32)
        obs["player_pose"] = obs["player_pose"].astype(np.float32)
        obs["boss_pose"] = obs["boss_pose"].astype(np.float32)
        obs["camera_pose"] = obs["camera_pose"].astype(np.float32)
        return obs

    @property
    def info(self) -> Dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        return {"allowed_actions": self.current_valid_actions()}

    def _env_setup(self, init_retries: int = 3):
        """Execute the Iudex environment setup.

        Args:
            init_retries: Maximum number of retries in case of initialization failure.

        Raises:
            RetriesExceededError: Setup failed more than ``init_retries`` times.
        """
        self._env_setup_init_check()
        while not self._env_setup_check() and init_retries >= 0:
            if not init_retries:
                logger.error("Maximum number of teleport resets exceeded")
                raise RetriesExceededError("Iudex environment setup failed")
            init_retries -= 1
            self._iudex_setup()
        self._is_init = True
        self.game.pause_game()

    def reset(self, seed: int | None = None, options: Any | None = None) -> Tuple[dict, dict]:
        """Reset the environment to the beginning of an episode.

        Args:
            seed: Random seed. Required by gymnasium, but does not apply to SoulsGyms.
            options: Options argument required by gymnasium. Not used in SoulsGym.

        Returns:
            A tuple of the first game state and the info dict after the reset.
        """
        if not self._is_init:
            self._env_setup()
        self.terminated = False
        self._game_input.reset()
        self.game.pause_game()
        self.game.allow_attacks = False
        self.game.allow_hits = False
        self.game.allow_moves = False
        self.game.reset_player_hp()
        self.game.reset_player_sp()
        self.game.time = 0  # Reset the total play time to avoid stuck timer on 999.99h
        if self.phase == 1:
            self.game.reset_boss_hp("iudex")
        # If Iudex is set to phase 2, set HP to 100 to trigger phase transition and wait
        elif not self._phase_init:
            self._phase_2_setup()
        self.game.game_speed = 3  # Speed up recovery
        if self._init_pose_randomization:
            # Sample player pose uniformly at random around Iudex
            player_pose = random.choice(
                self.game.data.coordinates[self.ENV_ID]["player_init_poses_random"])
        else:
            player_pose = self.game.data.coordinates[self.ENV_ID]["player_init_pose"]
        # When the arena is reset in between episodes, the player is possibly locked on to Iudex or
        # a unit outside the arena. This changes the player's orientation and prevents the
        # successful teleport to the initial pose. We therefore have to release lock on
        if self.game.lock_on:
            self._game_input.single_action("lockon", 0.005)
        tstart = time.time()
        while not self._reset_check(player_pose):
            self.game.player_pose = player_pose
            self.game.iudex_pose = self.game.data.coordinates[self.ENV_ID]["boss_init_pose"]
            # On rare occasions, the player can still get stuck with log ons outside the arena, or
            # races lead to unexpected bugs. In that case, we completely reset the environment
            if not self._reset_inner_check() or time.time() - tstart > 10:
                self.game.reload()
                self._env_setup()
                return self.reset()
            self.game.sleep(0.01)
        while not self.game.lock_on:
            self._lock_on()
            self._game_input.update_input()
            self.game.sleep(0.01)
            self._game_input.reset()  # Prevent getting stuck if initial press is missed
        self.game.pause_game()
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_moves = True
        self._internal_state = self.game.get_state(self.ENV_ID, use_cache=True)
        return self.obs, self.info

    def _iudex_setup(self) -> None:
        """Set up Iudex flags, focus the application, teleport to the fog gate and enter."""
        self.game.game_speed = 3  # In case SoulsGym crashed without unpausing Dark Souls III
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_iudex_setup: Reload due to incorrect boss flags")
            self.game.set_boss_flags("iudex", True)
            self.game.reload()
            logger.debug("_iudex_setup: Player respawn success")
        # Make sure to start around the bonfire. Also, in case the player has entered the arena on a
        # previous try, Iudex has to deaggro before the player can enter the arena again. This
        # forces us to reload
        d_pos = self.game.player_pose[:3] - self.game.data.coordinates[self.ENV_ID]["bonfire"][:3]
        if np.linalg.norm(d_pos) > 10:
            logger.debug("_iudex_setup: Reload due to incorrect player position")
            self.game.reload()
            logger.debug("_iudex_setup: Player respawn success")
        logger.debug("_iudex_setup: Reset success")
        self._game_window.focus_application()
        logger.debug("_iudex_setup: Focus success")
        # At this point the player should be idle. If this is not the case for some reason,
        # preemtively reload the game
        if not self.game.player_animation == "Idle":
            logger.debug("_iudex_setup: Reload due to unexpected player animation")
            self.game.reload()
            logger.debug("_iudex_setup: Player respawn success")
        self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["fog_wall"]
        self.game.sleep(0.2)
        d_pos = self.game.player_pose[:3] - self.game.data.coordinates[self.ENV_ID]["fog_wall"][:3]
        if np.linalg.norm(d_pos) > 0.1:
            logger.debug("_iudex_setup: Teleport failed. Retrying")
            self.game.reload()
            return self._iudex_setup()
        self._enter_fog_gate()
        self._phase_init = False
        logger.debug("_iudex_setup: Done")

    def _phase_2_setup(self):
        """Reduce Iudex HP to trigger phase transition and wait for completion."""
        self.game.iudex_hp = 100
        self.game.allow_attacks = True  # Iudex needs to attack for the transition
        self.game.game_speed = 3
        while not self.game.iudex_animation == "Attack1500":
            self.game.sleep(0.1)
        while self.game.iudex_animation == "Attack1500":
            self.game.sleep(0.1)
        self.game.allow_attacks = False
        self.game.pause_game()
        self.game.reset_boss_hp("iudex")
        self._phase_init = True

    def _enter_fog_gate(self):
        """Enter the fog gate."""
        self.game.camera_pose = self.env_args.cam_setup_orient
        self._game_input.single_action("interact")
        while True:
            if self.game.player_animation == "Idle":
                break
            self.game.sleep(0.1)
        fog_wall_pos = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"][:3]
        if np.linalg.norm(self.game.player_pose[:3] - fog_wall_pos) > 0.1:
            return  # Player has not entered the fog wall, abort early
        logger.debug("_enter_fog_gate: Done")

    @staticmethod
    def compute_reward(game_state: GameState, next_game_state: GameState) -> float:
        """Compute the reward from the current game state and the next game state.

        Args:
            game_state: The game state before the step.
            next_game_state: The game state after the step.

        Returns:
            The reward for the provided game states.
        """
        boss_reward = (game_state.boss_hp - next_game_state.boss_hp) / game_state.boss_max_hp * 100.
        player_hp_diff = (next_game_state.player_hp - game_state.player_hp)
        player_reward = player_hp_diff / game_state.player_max_hp * 100.
        if next_game_state.boss_hp == 0 or next_game_state.player_hp == 0:
            base_reward = 10 if next_game_state.boss_hp == 0 else -10
        else:
            # Experimental: Reward for moving towards the arena center, no reward within 4m distance
            d_center_now = np.linalg.norm(next_game_state.player_pose[:2] - np.array([139., 596.]))
            d_center_prev = np.linalg.norm(game_state.player_pose[:2] - np.array([139., 596.]))
            base_reward = -0.1 + 10 * (d_center_prev - d_center_now) * (d_center_now > 4)
        return boss_reward + player_reward + base_reward

    def _reset_check(self, player_pose: np.ndarray) -> bool:
        """Check if the environment reset was successful.

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

    def _reset_inner_check(self) -> bool:
        """Check if a critical event has happened during the inner reset loop.

        Returns:
            True if no problem has been detected, else False.
        """
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_reset_inner_check failed: Iudex flags not set properly")
            return False
        # Make sure the player and Iudex are still within arena bounds
        coords = zip(self.env_args.coordinate_box_low, self.game.player_pose,
                     self.env_args.coordinate_box_high)
        if not all(low < pos < high for low, pos, high in coords):
            logger.debug("_reset_inner_check failed: Player position out of arena bounds")
            logger.debug(self.game.get_state(self.ENV_ID))
            return False
        coords = zip(self.env_args.coordinate_box_low, self.game.iudex_pose,
                     self.env_args.coordinate_box_high)
        if not all(low < pos < high for low, pos, high in coords):
            logger.debug("_reset_inner_check failed: Iudex position out of arena bounds")
            logger.debug(self.game.get_state(self.ENV_ID))
            return False
        # Player and Iudex have to be alive
        if self.game.player_hp <= 0:
            logger.debug("_reset_check failed: Player HP below 1")
            return False
        if self.game.iudex_hp <= 0:
            logger.debug("_reset_check failed: Boss HP below 1")
            return False
        return True

    def _env_setup_init_check(self):
        """Check if all conditions for starting the environment setup are met.

        Raises:
            GameStateError: Game does not seem to be open.
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        try:
            game_state = self.game.get_state(self.ENV_ID)
        except MemoryReadError:
            logger.error("_env_setup_init_check failed: Player does not seem to be ingame")
            raise GameStateError("Player does not seem to be ingame")
        if game_state.player_animation != "Idle":
            self.game.reload()

    def _env_setup_check(self) -> bool:
        """Check if the environment setup was successful.

        Returns:
            True if all conditions are met, else False.
        """
        post_fog_wall_pos = self.game.data.coordinates[self.ENV_ID]["post_fog_wall"][:3]
        if np.linalg.norm(self.game.player_pose[:3] - post_fog_wall_pos) > 0.2:
            logger.debug("_env_setup_check failed: Player pose out of tolerances")
            return False
        if self.game.player_hp == 0:
            logger.debug("_env_setup_check failed: Player HP is 0")
            return False
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_env_setup_check failed: Incorrect boss flags")
            return False
        if self.game.player_animation != "Idle":
            logger.debug(f"_env_setup_check: Unexpected animation {self.game.player_animation}")
        return True


class IudexImgEnv(IudexEnv):

    def __init__(self, game_speed: int = 1., phase: int = 1, init_pose_randomization: bool = False):
        super().__init__(game_speed, phase, init_pose_randomization)
        self.observation_space = spaces.Box(low=0, high=255, shape=(90, 160, 3), dtype=np.uint8)

    @property
    def obs(self) -> np.ndarray:
        """Return the current observation."""
        return self.game.img

    @property
    def info(self) -> Dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        return {
            "allowed_actions": self.current_valid_actions(),
            "boss_hp": self._internal_state.boss_hp
        }


class IudexEnvDemo(SoulsEnvDemo, IudexEnv):
    """Demo environment for the Iudex Gundyr fight.

    Covers both phases. Player and boss loose HP, and the episode does not reset.
    """

    def __init__(self, game_speed: int = 1., init_pose_randomization: bool = False):
        """Initialize the demo environment.

        Args:
            game_speed: Determines how fast the game runs during :meth:`.SoulsEnv.step`.
            init_pose_randomization: Flag to randomize the player pose on reset.
        """
        super().__init__(game_speed)
        # IudexEnv can't be called with all arguments, so we have to set it manually after __init__
        self._init_pose_randomization = init_pose_randomization
        self.phase = 1

    def reset(self, seed: int | None = None, options: Any | None = None) -> Tuple[dict, dict]:
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

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
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
        if self._internal_state.boss_animation == "Attack1500":  # Phase change animation
            self.phase = 2
        obs["phase"] = self.phase
        return obs, reward, terminated, truncated, info
