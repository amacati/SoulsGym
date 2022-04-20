"""Logger for the internal game state."""
from soulsgym.core.game_interface import Game
from soulsgym.core.game_state import GameState


class Logger:
    """Snapshot all relevant ingame states and provide them as a :class:`~soulsgym.core.game_state.GameState` log."""

    def __init__(self, boss_id: str):
        """Initialize the game state with static attributes.

        Args:
            boss_id: Specifies which boss is logged.
        """
        self.game = Game()
        self.boss_id = boss_id
        self._log = GameState(player_max_hp=self.game.player_max_hp,
                              player_max_sp=self.game.player_max_sp,
                              boss_max_hp=self.game.get_boss_max_hp(boss_id))

    def log(self) -> GameState:
        """Read the current game state.

        Returns:
            A copy of the current GameState.
        """
        self._log.lock_on = self.game.lock_on
        self._log.boss_pose = self.game.get_boss_pose(self.boss_id)
        self._log.boss_hp = self.game.get_boss_hp(self.boss_id)
        self._boss_animation_task()  # Animations need special treatment
        self._log.player_animation = self.game.player_animation
        self._log.player_pose = self.game.player_pose
        self._log.camera_pose = self.game.camera_pose
        self._log.player_hp = self.game.player_hp
        self._log.player_sp = self.game.player_sp
        return self._log.copy()

    def _boss_animation_task(self):
        animation_name = self.game.get_boss_animation(self.boss_id)
        self._log.phase = 1 if self._log.phase == 1 and animation_name != "Attack1500" else 2
        if "Attack" in animation_name or "Atk" in animation_name:
            animation_name += "_P2" if self._log.phase == 2 else "_P1"
        self._log.boss_animation = animation_name
