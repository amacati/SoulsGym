.. _envs.margit:

Margit
======

.. automodule:: soulsgym.envs.eldenring.margit

Notes
-----

- The environment ID is ``SoulsGymMargit-v0``.
- The game window has to use the ``800x450`` capture resolution. Pass ``game_resolution`` to the
  environment to use another resolution.
- The observation contains the IDs of the animation names that are defined in the static data.
  Animations that are not defined are observed as ``-1``. See :ref:`games.eldenring` for details.

Known limitations
-----------------

- Only phase 1 is supported. Creating the environment with ``phase=2`` raises
  ``NotImplementedError``. Accordingly, only the animations of phase 1 are defined.
- The debug flags of Elden Ring are not mapped. The environment therefore does not freeze the
  entities while it resets the poses, and it cannot prevent player deaths. A death in the game,
  e.g. from a fall out of the arena, ends the episode, and the next reset reloads the game.
- The arena limits are not calibrated yet. The environment does not detect a player that leaves
  the arena or falls before the player dies.
