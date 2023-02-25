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

from pymem.exception import MemoryReadError
import numpy as np
from gym import spaces
from gym.error import RetriesExceededError

from soulsgym.envs.soulsenv import SoulsEnv, SoulsEnvDemo
from soulsgym.core.game_state import GameState
from soulsgym.exception import GameStateError, InvalidPlayerStateError
from soulsgym.core.static import boss_animations, player_animations, coordinates

logger = logging.getLogger(__name__)


class IudexEnv(SoulsEnv):
    """``IudexEnv`` implements the environment of the Iudex Gundyr boss fight."""

    ENV_ID = "iudex"

    def __init__(self, use_info: bool = False, phase: int = 1):
        """Define the state space.

        Args:
            use_info: Turns on additional information via the ``info`` return values in ``step``.
            phase: Set the boss phase. Either 1 or 2 for Iudex.
        """
        super().__init__(use_info=use_info)  # DarkSoulsIII needs to be open at this point
        # The state space consists of multiple spaces. These represent:
        # 1) Boss phase. Either 1 or 2 for Iudex
        # 2) Player and boss stats. In order: Player HP, Player SP, Boss HP
        # 3) Player, boss and camera poses. In order: Player x, y, z, a, boss x, y, z, a,
        #    camera x, y, z, nx, ny, nz where a represents the orientation and [nx ny nz] the camera
        #    plane normal
        # 4) Player animation
        # 5) Player animation duration. We assume no animation takes longer than 10s
        # 6) Boss animation
        # 7) Boss animation duration. We assume no animation takes longer than 10s.
        player_anim_len = len(player_animations["standard"]) + len(player_animations["critical"])
        stats_space = spaces.Box(np.array(self.env_args.space_stats_low, dtype=np.float32),
                                 np.array(self.env_args.space_stats_high, dtype=np.float32))
        pose_space = spaces.Box(np.array(self.env_args.space_coords_low, dtype=np.float32),
                                np.array(self.env_args.space_coords_high, dtype=np.float32))
        camera_pose_space = spaces.Box(
            np.array(self.env_args.space_coords_low + [-1, -1, -1], dtype=np.float32),
            np.array(self.env_args.space_coords_low + [1, 1, 1], dtype=np.float32))
        self.state_space = spaces.Dict({
            "phase": spaces.Discrete(2),
            "stats": stats_space,
            "player_pose": pose_space,
            "boss_pose": pose_space,
            "camera_pose": camera_pose_space,
            "player_animation": spaces.Discrete(player_anim_len),
            "player_animation_duration": spaces.Box(0., 10., (1,)),
            "boss_animation": spaces.Discrete(len(boss_animations["iudex"]["all"])),
            "boss_animation_duration": spaces.Box(0., 10., (1,))
        })
        assert phase in (1, 2)
        self.phase = phase
        self._phase_init = False

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

    def reset(self) -> GameState:
        """Reset the environment to the beginning of an episode.

        Returns:
            The first game state after a reset.
        """
        if not self._is_init:
            self._env_setup()
        self.done = False
        self._game_input.reset()
        self.game.pause_game()
        self.game.allow_attacks = False
        self.game.allow_hits = False
        self.game.allow_moves = False
        self.game.reset_player_hp()
        self.game.reset_player_sp()
        if self.phase == 1:
            self.game.reset_boss_hp("iudex")
        # If Iudex is set to phase 2, set HP to 100 to trigger phase transition and wait
        elif not self._phase_init:
            self.game.iudex_hp = 100
            self.game.allow_attacks = True  # Iudex needs to attack for the transition
            self.game.resume_game()
            while not self.game.iudex_animation == "Attack1500":
                self.game.sleep(0.1)
            while self.game.iudex_animation == "Attack1500":
                self.game.sleep(0.1)
            self.game.allow_attacks = False
            self.game.pause_game()
            self.game.reset_boss_hp("iudex")
            self._phase_init = True

        self.game.global_speed = 3  # Speed up recovery
        while not self._reset_check():
            self.game.player_pose = coordinates["iudex"]["player_init_pose"]
            self.game.iudex_pose = coordinates["iudex"]["boss_init_pose"]
            if not self._reset_inner_check():
                self.game.reload()
                self._env_setup()
                return self.reset()
            self.game.sleep(0.01)
        while not self.game.lock_on:
            self._lock_on(self.game.iudex_pose[:3])
            self._game_input.update_input()
        self.game.pause_game()
        self.game.allow_attacks = True
        self.game.allow_hits = True
        self.game.allow_moves = True
        self._internal_state = self._game_logger.log()
        return self._internal_state

    def _iudex_setup(self) -> None:
        """Set up Iudex flags, focus the application, teleport to the fog gate and enter."""
        self.game.resume_game()  # In case SoulsGym crashed without unpausing Dark Souls III
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_iudex_setup: Reload due to incorrect boss flags")
            self.game.set_boss_flags("iudex", True)
            self.game.reload()
            logger.debug("_iudex_setup: Player respawn success")
        # Make sure to start around the bonfire. Also, in case the player has entered the arena on a
        # previous try, Iudex has to deaggro before the player can enter the arena again. This
        # forces us to reload
        if np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["bonfire"][:3]) > 10:
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
        self.game.player_pose = coordinates["iudex"]["fog_wall"]
        self.game.sleep(0.2)
        if np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["fog_wall"][:3]) > 0.1:
            logger.debug("_iudex_setup: Teleport failed. Retrying")
            self.game.reload()
            return self._iudex_setup()
        self._enter_fog_gate()
        logger.debug("_iudex_setup: Done")

    def _enter_fog_gate(self):
        """Enter the fog gate."""
        self.game.camera_pose = self.env_args.cam_setup_orient
        self._game_input.single_action("interact")
        while True:
            if self.game.player_animation == "Idle":
                break
            self.game.sleep(0.1)
        dist = np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["post_fog_wall"][:3])
        if dist > 0.1:
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
            d_center_prev = np.linalg.norm(next_game_state.player_pose[:2] - np.array([139., 596.]))
            base_reward = -0.1 + 0.1 * (d_center_prev - d_center_now) * (d_center_now > 4)
        return boss_reward + player_reward + base_reward

    def _reset_check(self) -> bool:
        """Check if the environment reset was successful.

        Returns:
            True if the reset was successful, else False.
        """
        dist = np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["player_init_pose"][:3])  # noqa: E501, yapf: disable
        if dist > 1:
            return False
        dist = np.linalg.norm(self.game.iudex_pose[:3] - coordinates["iudex"]["boss_init_pose"][:3])
        if dist > 1:
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
        bounds = (low < pos < high for low, pos, high in zip(
            self.env_args.space_coords_low, self.game.player_pose, self.env_args.space_coords_high))
        if not all(bounds):
            logger.debug("_reset_inner_check failed: Player position out of arena bounds")
            logger.debug(self._game_logger.log())
            return False
        bounds = (low < pos < high for low, pos, high in zip(
            self.env_args.space_coords_low, self.game.iudex_pose, self.env_args.space_coords_high))
        if not all(bounds):
            logger.debug("_reset_inner_check failed: Iudex position out of arena bounds")
            logger.debug(self._game_logger.log())
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
            game_state = self._game_logger.log()
        except MemoryReadError:
            logger.error("_env_setup_init_check failed: Player does not seem to be ingame")
            raise GameStateError("Player does not seem to be ingame")
        if game_state.player_animation != "Idle":
            logger.error("_env_setup_init_check failed: Player is not idle")
            logger.error(game_state)
            raise InvalidPlayerStateError("Player is not idle")

    def _env_setup_check(self) -> bool:
        """Check if the environment setup was successful.

        Returns:
            True if all conditions are met, else False.
        """
        dist = np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["post_fog_wall"][:3])
        if dist > 0.2:
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


class IudexEnvDemo(SoulsEnvDemo, IudexEnv):
    """Demo environment for the Iudex Gundyr fight.

    Covers both phases. Player and boss loose HP, and the episode does not reset.
    """

    def __init__(self, use_info: bool = False):
        """Initialize the demo environment.

        Args:
            use_info: Turns on additional information via the ``info`` return values in ``step``.
        """
        super().__init__(use_info=use_info)
