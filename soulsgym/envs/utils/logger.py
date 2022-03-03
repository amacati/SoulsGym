from __future__ import annotations
from typing import Any, List
import threading
from dataclasses import dataclass

import numpy as np
from nptyping import NDArray

import soulsgym.envs.utils.game_interface as game
from soulsgym.envs.utils.tables import p1_anim_enc

Position = NDArray[np.float32]


@dataclass
class GameState:
    """
    Represents a snapshot-state of the game.
    """
    phase: int = 1
    player_hp: int = 0
    player_max_hp: int = 0
    player_sp: int = 0
    player_max_sp: int = 0
    boss_hp: int = 0
    boss_max_hp: int = 0
    player_pos: Position = np.zeros(4, dtype=np.float32)
    boss_pos: Position = np.zeros(4, dtype=np.float32)
    animation: str = "NoAnimation"
    player_animation: str = "NoAnimation"
    animation_count: int = 0
    locked_on: bool = False
    iudex_def: bool = False

    def copy(self) -> GameState:
        "Returns a copy of itself."
        return GameState(**self.__dict__)

    def toarray(self) -> List:
        """
        Converts the state to a usable input for agent networks.

        Contains normalized player hp, sp, boss hp, player x,y,a position and boss x, y, a position.
        Current animation is one-hot encoded at the end of the array.

        Returns:
            The converted state as list.
        """
        x = [
            self.player_hp / self.player_max_hp, self.player_sp / self.player_max_sp,
            self.boss_hp / self.boss_max_hp, self.player_x, self.player_y, self.player_a,
            self.boss_x, self.boss_y, self.boss_a
        ]
        x.extend(p1_anim_enc.transform(self.animation))
        x.append(self.animation_count)
        return x

    @staticmethod
    def state_size() -> int:
        """
        Returns the size of a GameState converted by GameState.toarray.
        """
        return 28

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        setattr(self, name, value)


class Logger:
    """
    Logger class used to abstract information, collected from the game.
    """

    def __init__(self):
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
        """
        Reads the current game state and returns the log as a GameState.
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
        self._log.boss_hp = game.get_targeted_hp()
        self._log.boss_max_hp = game.get_targeted_max_hp()  # Target might change

    def _boss_def_task(self):
        self._log.iudex_def = game.get_iudex_defeated()

    def _player_pos_task(self):
        self._log.player_pos = game.get_player_position()

    def _boss_pos_task(self):
        self._log.boss_pos = game.get_targeted_position()

    def _player_anim_task(self):
        animation = game.get_player_animation()
        self._log.player_animation = animation

    def _boss_anim_task(self):
        animation_name = game.get_targeted_animation()
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
    """
    A helper class that calls a task every time ping is called and waits until complition on
    wait_for. This object starts its own thread for this.
    """

    def __init__(self, task):
        self.task = task
        self._task_lock = threading.Lock()
        self._join_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ping(self):
        """
        Pings the ThreadBoost object to perform its task
        """
        self._task_lock.release()

    def wait_for(self):
        """
        Blocking call, waiting for the completion of the task.
        """
        self._join_lock.acquire()

    def _run(self):
        self._task_lock.acquire()
        self._join_lock.acquire()
        while True:
            self._task_lock.acquire()
            self.task()
            self._join_lock.release()
