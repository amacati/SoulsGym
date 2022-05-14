"""The ``Logger`` logs the current game state into :class:`.GameState` instances.

Note:
    We do not have access to the length of the current animation and are unable to determine the
    animation count in the logger. All animation counts in the ``GameState`` are set to 0. The
    animation count tracking is the responsibility of the gyms instead.

Todo:
    * Include a phase detection independent of the transition animation ``Attack1500``.
"""
from soulsgym.core.game_interface import Game
from soulsgym.core.game_state import GameState


class Logger:
    """The ``Logger`` snapshots all relevant game states into a ``GameState``."""

    def __init__(self, boss_id: str):
        """Initialize the game state with static attributes.

        Args:
            boss_id: The target boss ID.
        """
        self.game = Game()
        self.boss_id = boss_id
        self._game_state = GameState(player_max_hp=self.game.player_max_hp,
                                     player_max_sp=self.game.player_max_sp,
                                     boss_max_hp=self.game.get_boss_max_hp(boss_id))

    def log(self) -> GameState:
        """Read the current game state.

        Returns:
            A copy of the current :class:`.GameState`.
        """
        self._game_state.lock_on = self.game.lock_on
        self._game_state.boss_pose = self.game.get_boss_pose(self.boss_id)
        self._game_state.boss_hp = self.game.get_boss_hp(self.boss_id)
        self._game_state.boss_animation = self.game.get_boss_animation(self.boss_id)
        self._game_state.player_animation = self.game.player_animation
        self._game_state.player_pose = self.game.player_pose
        self._game_state.camera_pose = self.game.camera_pose
        self._game_state.player_hp = self.game.player_hp
        self._game_state.player_sp = self.game.player_sp
        return self._game_state.copy()
