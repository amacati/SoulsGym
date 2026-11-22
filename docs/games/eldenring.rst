.. _games.eldenring:

Elden Ring
==========

.. automodule:: soulsgym.games.eldenring

Known constraints
-----------------

- The debug flags of Elden Ring are not mapped to the game memory yet. In contrast to Dark Souls
  III, the game interface therefore does not provide the flags to disable attacks, hits, moves and
  deaths, or the lock on range settings. The environments do not freeze the entities while they
  reset the poses.
- Elden Ring reports animations as integer animation IDs. The animation names of Margit are defined
  in ``animations.yaml``, and IDs that are not defined there are reported as ``Unknown_<ID>``. Only
  the animations of phase 1 are defined so far. The player animations are defined by their IDs and
  their timings have not been measured yet.

Game settings
-------------

- Elden Ring uses the key mapping in ``soulsgym/core/data/eldenring/keys.yaml``. Set these key
  bindings in System > Keyboard/Mouse Settings > Key Bindings:

  .. list-table::
     :widths: 60 30
     :header-rows: 1

     * - Action
       - Keyboard
     * - Move Forwards / Backwards / Left / Right
       - W / S / A / D
     * - Backstep, Dodge Roll, Dash
       - Space
     * - Move Camera / Change Target (Up / Down / Left / Right)
       - J / K / O / P
     * - Reset Camera, Lock-On/Remove Target
       - Q
     * - Attack (RH & Two-Handed Armament)
       - L
     * - Strong Attack (RH & 2H Armament)
       - Y
     * - Skill
       - M
     * - Use Item
       - R
     * - Event Action (Examine, Open, etc.)
       - E

  SoulsGym only sends keyboard keys, so the attacks need a keyboard binding in addition to the
  mouse. Remove these keys from all other actions (for example Jump or Map), otherwise an action
  can trigger a second one or open a menu.

- The game has to run in offline mode without Easy Anti-Cheat. SoulsGym cannot attach to the game
  process while Easy Anti-Cheat is active. Exclusively use a dedicated save game with SoulsGym, as
  described in the ban protection section of :ref:`setup`.
