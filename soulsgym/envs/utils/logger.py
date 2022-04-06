"""Logger for the internal game state."""
from soulsgym.envs.utils.game_interface import Game
from soulsgym.envs.utils.gamestate import GameState


class Logger:
    """Snapshot all relevant ingame states and provide them as a `GameState` log."""

    def __init__(self):
        """Initialize logging threads."""
        self._log = None
        self.game = Game()
        # Create threads for every kind of task, to read data more quickly
        # Tread order matters! First threads are without targeted_entity_info necessary
        self.tasks = [
            self._locked_on_task,
            self._player_pos_task,
            self._player_stats_task,
            self._player_anim_task,
            self._boss_pos_task,
            self._boss_hp_task,
            self._boss_anim_task,
        ]

    def log(self, no_target: bool = False) -> GameState:
        """Read the current game state.

        Args:
            no_target: Switch off target tasks which can crash if no target was locked prior to the
                call.

        Returns:
            A copy of the current GameState.
        """
        if self._log is None:
            self._log = GameState(player_max_hp=self.game.player_max_hp,
                                  player_max_sp=self.game.player_max_sp)
        self._locked_on_task()
        if not no_target:
            self._boss_pos_task()
            self._boss_hp_task()
            self._boss_anim_task()
        self._player_anim_task()
        self._player_pos_task()
        self._player_stats_task()
        return self._log.copy()

    def _player_stats_task(self):
        self._log.player_hp = self.game.player_hp
        self._log.player_sp = self.game.player_sp

    def _boss_hp_task(self):
        self._log.boss_hp = self.game.target_hp
        self._log.boss_max_hp = self.game.target_max_hp  # Target might change

    def _player_pos_task(self):
        self._log.player_pos = self.game.player_position

    def _boss_pos_task(self):
        self._log.boss_pos = self.game.target_position

    def _player_anim_task(self):
        self._log.player_animation = self.game.player_animation

    def _boss_anim_task(self):
        anim_name = self.game.target_animation
        self._log.phase = 1 if self._log.phase == 1 and anim_name != "Attack1500" else 2
        if "Attack" in anim_name or "Atk" in anim_name:
            anim_name = anim_name + "_P2" if self._log.phase == 2 else anim_name + "_P1"
        self._log.boss_animation = anim_name

    def _locked_on_task(self):
        self._log.locked_on = self.game.get_locked_on()
