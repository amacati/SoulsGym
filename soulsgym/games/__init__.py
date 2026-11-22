"""This module contains the game interfaces for the Souls games."""

from soulsgym.games.game import Game  # noqa: TC001, I001, required to avoid circular import errors
from soulsgym.games.darksouls3 import DarkSoulsIII
from soulsgym.games.eldenring import EldenRing


def game_factory(game_id: str, game_resolution: tuple[int, int] | None = None) -> Game:
    """Factory function for creating game interfaces.

    Args:
        game_id: The name of the game.
        game_resolution: Optional Elden Ring game-window resolution override.

    Returns:
        The game interface.
    """
    match game_id:
        case "DarkSoulsIII":
            return DarkSoulsIII()
        case "EldenRing":
            return EldenRing(game_resolution=game_resolution)
    raise ValueError(f"Unknown game: {game_id}")
