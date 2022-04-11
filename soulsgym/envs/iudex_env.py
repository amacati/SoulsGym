"""SoulsGym environment for Iudex Gundyr."""
import logging
import time

from pymem.exception import MemoryReadError
import numpy as np
from gym import spaces
from gym.error import RetriesExceededError

from soulsgym.envs.soulsenv import SoulsEnv
from soulsgym.core.game_state import GameState
from soulsgym.exception import GameStateError, InvalidPlayerStateError
from soulsgym.core.static import boss_animations, player_animations, coordinates

logger = logging.getLogger(__name__)


class IudexEnv(SoulsEnv):
    """SoulsEnv implementation for Iudex Gundyr."""

    ENV_ID = "iudex"

    def __init__(self):
        """Define the state/action spaces and initialize the game interface."""
        super().__init__()  # DarkSoulsIII needs to be open at this point
        # The state space consists of multiple spaces. These represent:
        # 1) Boss phase. Either 1 or 2 for Iudex
        # 2) Player and boss stats. In order: Player HP, Player SP, Boss HP
        # 3) Player and boss coordinates. In order: Player x, y, z, a, boss x, y, z, a, where a
        #    represents the orientation
        # 4) Player animation
        # 5) Boss animation
        # 6) Boss animation duration (in 0.1s ticks). We assume no animation takes longer than 10s
        p_anim_len = len(player_animations["standard"]) + len(player_animations["critical"])
        stats_space = spaces.Box(np.array(self.env_args.space_stats_low, dtype=np.float32),
                                 np.array(self.env_args.space_stats_high, dtype=np.float32))
        coords_space = spaces.Box(np.array(self.env_args.space_coords_low, dtype=np.float32),
                                  np.array(self.env_args.space_coords_high, dtype=np.float32))
        self.state_space = spaces.Dict({
            "phase": spaces.Discrete(2),
            "stats": stats_space,
            "coords": coords_space,
            "player_animation": spaces.Discrete(len(boss_animations["iudex"])),
            "boss_animation": spaces.Discrete(p_anim_len),
            "boss_animation_counter": spaces.Discrete(100)
        })

    def _env_setup(self, init_retries: int = 3):
        """Execute the Iudex environment setup.

        Args:
            init_retries: Maximum number of retries in case of initialization failure.

        Raises:
            RetriesExceededError: Setup failed more than `init_retries` times.
        """
        self._env_setup_check()
        while not self._reset_check(self._game_logger.log()) and init_retries >= 0:
            if not init_retries:
                logger.error("Maximum number of teleport resets exceeded")
                raise RetriesExceededError("Iudex environment setup failed")
            init_retries -= 1
            self._iudex_setup()
            dist = np.linalg.norm(coordinates["iudex"]["fog_wall"][:3] - self.game.player_pose[:3])
            if not dist < 1:
                continue  # Teleport not successful, reset again
            if self.game.player_hp == 0:
                continue  # Player has died on teleport
            self._initial_key_sequence()
        self.game.pause_game()

    def reset(self) -> GameState:
        """Reset the environment to the beginning of an episode.

        Returns:
            The first observation after a reset.
        """
        self.done = False
        self._game_input.reset()
        self.game.pause_game()
        self.game.clear_cache()  # Todo: remove
        player_init_pose = coordinates["iudex"]["player_init_pose"]
        iudex_init_pose = coordinates["iudex"]["boss_init_pose"]
        game_log = self._game_logger.log()
        self.game.iudex_attacks = False
        # Wait for the teleport to take effect. This ensures lock on is possible in the next steps
        player_distance = np.linalg.norm(self.game.player_pose[:3] - player_init_pose[:3])
        iudex_distance = np.linalg.norm(self.game.iudex_pose[:3] - iudex_init_pose[:3])
        while player_distance > 1 or iudex_distance > 1:
            if not self._reset_position():
                self.game.reload()
                self._env_setup()
                return self.reset()
            player_distance = np.linalg.norm(self.game.player_pose[:3] - player_init_pose[:3])
            iudex_distance = np.linalg.norm(self.game.iudex_pose[:3] - iudex_init_pose[:3])
        self.game.reset_player_hp()
        self.game.reset_player_sp()
        self.game.reset_boss_hp("iudex")
        self.game.weapon_durability = 70  # 70 is maximum durability
        while "Walk" not in game_log.boss_animation or game_log.player_animation != "Idle":
            if not self._reset_position():
                self.game.reload()  # Guarantees player is in Idle mode at the bonfire
                self._env_setup()
                return self.reset()
            if not game_log.lock_on:
                self._lock_on(self.game.iudex_pose[:3])
                game_log.lock_on = True
        if not game_log.lock_on:
            self._lock_on(self.game.iudex_pose[:3])
            game_log.lock_on = True
        self.game.iudex_attacks = True
        self._internal_state = self._game_logger.log()
        assert self._reset_check(self._internal_state), f"Fail in GameState {self._internal_state}"
        return self._internal_state

    def _reset_position(self) -> bool:
        self.game.player_pose = coordinates["iudex"]["player_init_pose"]
        self.game.iudex_pose = coordinates["iudex"]["boss_init_pose"]
        time.sleep(0.01)  # Allow teleport to take effect
        game_log = self._game_logger.log()
        if not self._reset_inner_check(game_log):  # Make sure the player and Iudex are alive
            return False
        self.game.resume_game()
        time.sleep(0.1)  # Give animations time to finish if they block the teleport
        self.game.pause_game()
        return True

    def _iudex_setup(self) -> None:
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
        time.sleep(1)
        if np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["fog_wall"][:3]) > 0.1:
            logger.debug("_iudex_setup: Teleport failed. Retrying")
            self.game.reload()
            return self._iudex_setup()
        logger.debug("_iudex_setup: Done")

    def _initial_key_sequence(self):
        self.game.camera_pose = self.env_args.cam_setup_orient
        self._game_input.single_action("interact")
        while True:
            if self.game.player_animation == "Idle":
                break
            time.sleep(0.1)
        self.game.camera_pose = self.env_args.cam_setup_orient
        dist = np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["post_fog_wall"][:3])
        if dist > 0.1:
            return  # Player has not entered the fog wall, abort early
        self._game_input.single_action("forward", press_time=4.0)  # Enter the arena
        self._lock_on()
        logger.debug("_init_key_sequence: Done")

    def compute_reward(self, game_log: GameState) -> float:
        """Compute the reward from a game observation.

        Args:
            game_log: A game state.

        Returns:
            The reward for the provided game observation.
        """
        player_hp_penalty = 1 - game_log.player_hp / game_log.player_max_hp
        player_sp_penalty = 1 - game_log.player_sp / game_log.player_max_sp
        boss_hp_reward = 1 - game_log.boss_hp / game_log.boss_max_hp
        return boss_hp_reward - player_hp_penalty - 0.05 * player_sp_penalty

    def _reset_check(self, game_log: GameState, include_lock_on: bool = True) -> bool:
        """Check if the environment reset was successful.

        Args:
            game_log: Current game log.
            include_lock_on: Omit lock on checks if set to True.

        Returns:
            True if the reset was successful, else False.
        """
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_reset_check failed: Iudex flags not set properly")
            logger.debug(game_log)
            return False
        dist = np.linalg.norm(game_log.player_pose[:3] - coordinates["iudex"]["player_init_pose"][:3])  # noqa: E501, yapf: disable
        if dist > 2:
            logger.debug("_reset_check failed: Player position out of tolerances")
            logger.debug(game_log)
            return False
        if game_log.player_hp != game_log.player_max_hp:
            logger.debug("_reset_check failed: Player HP is not at maximum")
            logger.debug(game_log)
            return False
        if game_log.boss_hp != game_log.boss_max_hp:
            logger.debug("_reset_check failed: Boss HP is not at maximum")
            logger.debug(game_log)
            return False
        if include_lock_on and not game_log.lock_on:
            logger.debug("_reset_check failed: No lock on")
            logger.debug(game_log)
            return False
        if include_lock_on:
            logger.debug("_reset_check: Done")
        return True

    def _reset_inner_check(self, game_log: GameState) -> bool:
        """Check if a critical event has happened during the inner reset loop.

        Returns:
            True if no problem has been detected, else False.
        """
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_reset_inner_check failed: Iudex flags not set properly")
            return False
        # Make sure the player and Iudex are still within arena bounds
        bounds = (low < pos < high for low, pos, high in zip(
            self.env_args.space_coords_low, game_log.player_pose, self.env_args.space_coords_high))
        if not all(bounds):
            logger.debug("_reset_inner_check failed: Player position out of arena bounds")
            logger.debug(game_log)
            return False
        bounds = (low < pos < high for low, pos, high in zip(
            self.env_args.space_coords_low, game_log.boss_pose, self.env_args.space_coords_high))
        if not all(bounds):
            logger.debug("_reset_inner_check failed: Iudex position out of arena bounds")
            logger.debug(game_log)
            return False
        # Player and Iudex have to be alive
        if game_log.player_hp <= 0:
            logger.debug("_reset_check failed: Player HP below 1")
            return False
        if game_log.boss_hp <= 0:
            logger.debug("_reset_check failed: Boss HP below 1")
            return False
        return True

    def _env_setup_check(self):
        """Check if the environment setup was successful.

        Raises:
            GameStateError: Game state is outside of expected values.
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        try:
            game_log = self._game_logger.log()
        except MemoryReadError:
            logger.error("_env_setup_check failed. Player does not seem to be ingame")
            raise GameStateError("Player does not seem to be ingame")
        if game_log.player_animation != "Idle":
            logger.error("_env_setup_check failed. Player is not idle")
            logger.error(game_log)
            raise InvalidPlayerStateError("Player is not idle")
        if np.linalg.norm(self.game.player_pose[:3] - coordinates["iudex"]["bonfire"][:3]) > 30:
            logger.error("_env_setup_check failed. Player is not close to the bonfire")
            logger.error(game_log)
            raise InvalidPlayerStateError("Player is not close to the bonfire `Cemetry of Ash`")
