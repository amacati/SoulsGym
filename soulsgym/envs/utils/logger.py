"""Logger for the internal game state.

Todo:
    * Rework the logger to simplify the threading construct.
"""
import threading
from typing import Callable

import soulsgym.envs.utils.game_interface as game
from soulsgym.envs.utils.gamestate import GameState


class Logger:
    """Snapshot all relevant ingame states and provide them as a `GameState` log."""

    def __init__(self):
        """Initialize logging threads."""
        self._log = None
        # Create threads for every kind of task, to read data more quickly
        # Tread order matters! First threads are without targeted_entity_info necessary
        tasks = [
            self._locked_on_task, self._player_pos_task, self._player_stats_task,
            self._player_anim_task, self._boss_pos_task, self._boss_hp_task, self._boss_anim_task,
            self._boss_def_task
        ]
        self.threads = [ThreadBoost(task) for task in tasks]

    def log(self, no_target: bool = False) -> GameState:
        """Read the current game state.

        Logging works by pinging all threads once.

        Args:
            no_target: Switch off target tasks which can crash if no target was locked prior to the
                call.

        Returns:
            A log as a GameState.
        """
        if self._log is None:
            self._log = GameState(player_max_hp=game.get_player_max_hp(),
                                  player_max_sp=game.get_player_max_sp())
        # re-read all data
        if no_target:
            for thread in self.threads[:4]:  # Only read non target values
                thread.ping()
            for thread in self.threads[:4]:
                thread.wait_for()
            return self._log.copy()
        for thread in self.threads:
            thread.ping()  # Releases wait for lock, starts one iteration of the inner loop
        for thread in self.threads:
            thread.wait_for()
        return self._log.copy()

    def _player_stats_task(self):
        hp, sp = game.get_player_hp_sp()
        self._log.player_hp = hp
        self._log.player_sp = sp

    def _boss_hp_task(self):
        self._log.boss_hp = game.get_target_hp()
        self._log.boss_max_hp = game.get_target_max_hp()  # Target might change

    def _boss_def_task(self):
        self._log.iudex_def = game.get_iudex_defeated()

    def _player_pos_task(self):
        self._log.player_pos = game.get_player_position()

    def _boss_pos_task(self):
        self._log.boss_pos = game.get_target_position()

    def _player_anim_task(self):
        animation = game.get_player_animation()
        self._log.player_animation = animation

    def _boss_anim_task(self):
        animation_name = game.get_target_animation()
        self._log.phase = 1 if self._log.phase == 1 and animation_name != "Attack1500" else 2
        animation_name = animation_name + "_P2" if self._log.phase == 2 else animation_name + "_P1"

        if self._log.animation == animation_name:
            self._log.animation_count += 1
        else:
            self._log.animation_count = 0

        self._log.animation = animation_name

    def _locked_on_task(self):
        self._log.locked_on = game.get_locked_on()


class ThreadBoost:
    """A helper class that calls a task on notification and can await its completion."""

    def __init__(self, task: Callable):
        """Initialize the locks and task thread.

        Args:
            task: The target task.
        """
        self.task = task
        self._task_lock = threading.Lock()
        self._join_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ping(self):
        """Ping the ThreadBoost object to perform its task."""
        self._task_lock.release()

    def wait_for(self):
        """Block the call while waiting for the completion of the task."""
        self._join_lock.acquire()

    def _run(self):
        self._task_lock.acquire()
        self._join_lock.acquire()
        while True:
            self._task_lock.acquire()
            self.task()
            self._join_lock.release()
