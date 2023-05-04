from soulsgym.core.games.game import Game
from soulsgym.core.games.darksouls3 import DarkSoulsIII
from soulsgym.core.games.eldenring import EldenRing


def game_factory(game_id: str) -> Game:
    """Factory function for creating game interfaces.

    Args:
        game_id: The name of the game.

    Returns:
        The game interface.
    """
    match game_id.lower().replace(" ", ""):
        case "darksoulsiii":
            return DarkSoulsIII()
        case "eldenring":
            return EldenRing()
    raise ValueError(f"Unknown game: {game_id}")
