soulsgym.games
==============

.. automodule:: soulsgym.games

Games
~~~~~

The ``soulsgym.games`` module contains one game class for each supported game. The idea is to
use games just like Python objects, where changes to the game object are translated into the
necessary memory reads and writes.

.. code-block::
   :caption: Example of using games to reset the player HP.

       from soulsgym.games import DarkSoulsIII

        game = DarkSoulsIII()  # Game needs to be running at this point
        game.player_hp = game.player_max_hp  # Resets the HP in the actual game

Currently, the following games have been added to ``soulsgym``:  
  
* :ref:`Dark Souls III <games.darksouls3>`
* :ref:`Elden Ring <games.eldenring>` 

 

.. note::
    The games are not feature complete. ``DarkSoulsIII`` only supports a limited number of bosses so
    far, and ``EldenRing`` lacks all boss support.

.. toctree::
    :hidden:
    
    game
    darksouls3
    eldenring
