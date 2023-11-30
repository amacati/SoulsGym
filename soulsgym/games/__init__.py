from soulsgym.games.game import Game
from soulsgym.games.darksouls3 import DarkSoulsIII
from soulsgym.games.eldenring import EldenRing


def game_factory(game_id: str) -> Game:
    """Factory function for creating game interfaces.

    Args:
        game_id: The name of the game.

    Returns:
        The game interface.
    """
    match game_id:
        case "DarkSoulsIII":
            return DarkSoulsIII()
        case "EldenRing":
            return EldenRing()
    raise ValueError(f"Unknown game: {game_id}")
