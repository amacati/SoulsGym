"""SoulsGym environment for Iudex Gundyr."""
import logging
import time
from pymem.exception import MemoryReadError
import numpy as np
from gym import spaces
from gym.error import RetriesExceededError

from soulsgym.envs.soulsenv import ObsType, SoulsEnv
from soulsgym.envs.utils.gamestate import GameState
from soulsgym.exception import GameStateError, InvalidPlayerStateError
from soulsgym.envs.utils import distance
from soulsgym.envs.utils.tables import phase1_animations, player_animations, coordinates

logger = logging.getLogger("SoulsGym")


class IudexEnv(SoulsEnv):
    """SoulsEnv implementation for Iudex Gundyr."""

    ENV_ID = "iudex"

    def __init__(self):
        """Fefine the state/action spaces and initialize the game interface."""
        super().__init__()  # DarkSoulsIII needs to be open at this point
        # The state space consists of multiple spaces. These represent:
        # 1) Boss phase. Either 1 or 2 for Iudex
        # 2) Player and boss stats. In order: Player HP, Player SP, Boss HP
        # 3) Player and boss coordinates. In order: Player x, y, z, a, boss x, y, z, a, where a
        #    represents the orientation
        # 4) Player animation
        # 5) Boss animation
        # 6) Boss animation duration (in 0.1s ticks). We assume no animation takes longer than 10s
        # TODO: Replace phase1_animations with total animations
        p_anim_len = len(player_animations["interrupt"]) + len(player_animations["no_interrupt"])
        stats_space = spaces.Box(np.array(self.env_args.space_stats_low, dtype=np.float32),
                                 np.array(self.env_args.space_stats_high, dtype=np.float32))
        coords_space = spaces.Box(np.array(self.env_args.space_coords_low, dtype=np.float32),
                                  np.array(self.env_args.space_coords_high, dtype=np.float32))
        self.state_space = spaces.Dict({
            "phase": spaces.Discrete(2),
            "stats": stats_space,
            "coords": coords_space,
            "player_animation": spaces.Discrete(len(phase1_animations)),
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
        while not self._reset_check(self._game_logger.log(no_target=True)) and init_retries > 0:
            if not init_retries:
                logger.error("Maximum number of teleport resets exceeded")
                raise RetriesExceededError("Iudex environment setup failed")
            init_retries -= 1
            self._iudex_setup()
            if not distance(coordinates["iudex"], self.game.player_position, flat=False) < 1:
                continue  # Omit initial key sequence for speed
            if self.game.player_hp == 0:
                continue  # Player has died on teleport
            self._initial_key_sequence()
        self.game.pause_game()
        print(self._game_logger.log())  # TODO: Remove

    def reset(self) -> ObsType:
        """Reset the environment to the beginning of an episode.

        Returns:
            The first observation after a reset.
        """
        game_log = self._game_logger.log()
        if not self._reset_check(game_log) or True:
            self.game.player_pos = coordinates["player_init_pos"]
            self.game.target_pos = self.env_args.iudex_init_pos
            self.game.reset_player_hp()
            self.game.reset_player_sp()
            self.game.reset_target_hp()
            self.game.player_animation = "Idle"
            self.game.target_animation = "WalkFrontBattle_P1"
            self._lock_on()
        self._internal_state = self._game_logger.log()
        assert self._reset_check(self._internal_state)
        return self._internal_state

    def _iudex_setup(self):
        self.game.resume_game()  # In case SoulsGym crashed without unpausing Dark Souls III
        if not self.game.check_boss_flags("iudex"):
            self.game.set_boss_flags("iudex", True)
            self.game.reload()
        logger.debug("_iudex_setup: Reset success")
        # time.sleep(3)  # Wait until loading screen comes up to focus on the application
        self._game_window.focus_application()
        logger.debug("_iudex_setup: Focus success")
        self.game.clear_cache(
        )  # Reset memory manupulator cache after death has invalidated addresses
        while True:
            time.sleep(1)
            try:
                if self.game.player_animation in ("", "DeathIdle"):  # Game cache invalid at death
                    self.game.clear_cache()
                if self.game.player_animation == "Idle":
                    time.sleep(0.05)  # Give game time to get to a "stable" state
                    break
            except (MemoryReadError, UnicodeDecodeError):  # Read during death reset might fail
                continue
        logger.debug("_iudex_setup: Player respawn success")
        self.game.player_position = coordinates["iudex"]
        time.sleep(1)
        logger.debug("_iudex_setup: Success")

    def _initial_key_sequence(self):
        self._game_input.single_action("Interact")
        while True:
            if self.game.player_animation == "Idle":
                break
            time.sleep(0.1)
        self._game_input.single_action("Forward", press_time=4.0)
        self._game_input.single_action("LockOn")
        logger.debug("_init_key_sequence: Success")

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

    def _reset_check(self, game_log: GameState) -> bool:
        """Check if the environment reset was successful.

        Args:
            game_log: Current game log.

        Returns:
            True if the reset was successful, else False.
        """
        if not self.game.check_boss_flags("iudex"):
            logger.debug("_reset_check failed: Iudex flags not set properly")
            return False
        if distance(game_log.player_pos, coordinates["player_init_pos"], flat=False) > 2:
            logger.debug("_reset_check failed: Player position out of tolerances")
            return False
        if game_log.player_hp != game_log.player_max_hp:
            logger.debug("_reset_check failed: Player HP is not at maximum")
            return False
        if game_log.boss_hp != game_log.boss_max_hp:
            logger.debug("_reset_check failed: Boss HP is not at maximum")
            return False
        if not game_log.locked_on:
            logger.debug("_reset_check failed: No lock on")
            # return False  TODO: Reenable
        logger.debug("_reset_check successful")
        return True

    def _env_setup_check(self):
        """Check if the environment setup was successful.

        Raises:
            GameStateError: Game state is outside of expected values.
            InvalidPlayerStateError: Player state is outside of expected values.
        """
        try:
            game_log = self._game_logger.log(no_target=True)
        except MemoryReadError:
            raise GameStateError("Player does not seem to be ingame")
        if game_log.player_animation != "Idle":
            raise InvalidPlayerStateError("Player is not idle")
        if distance(self.game.player_position, self.env_args.p_setup_coords) > 30:
            raise InvalidPlayerStateError("Player is not close to the bonfire `Cemetry of Ash`")
        if game_log.player_hp != self.env_args.space_stats_high[0]:
            raise InvalidPlayerStateError("Player HP differs from expected value. Please make sure \
                to start with the correct stats for SoulsGym!")
