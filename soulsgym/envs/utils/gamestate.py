"""GameState data class for storing the internal game state."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any

import numpy as np
from nptyping import NDArray

from soulsgym.envs.utils.tables import p1_anim_enc

Position = NDArray[np.float32]


@dataclass
class GameState:
    """Represent a snapshot-state of the game."""

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
        """Copy the object.

        Returns:
            A copy of itself.
        """
        return GameState(**self.__dict__)

    def toarray(self) -> List:
        """Convert the state to a usable input for agent networks.

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

    def state_size(self) -> int:
        """Get the size of a GameState.

        Returns:
            The size of a GameState converted by GameState.toarray.
        """
        return len(self.toarray())

    def __getitem__(self, name: str) -> Any:
        """Enable attribute access by key indexing.

        Args:
            name: Attribute name.

        Returns:
            The attribute value.
        """
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any):
        """Enable attribute assignment by key indexing.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        setattr(self, name, value)
