.. _setup:

SoulsGym setup
==============

Requirements
~~~~~~~~~~~~
SoulsGym requires you to have Dark Souls III installed in version 1.15.2 with all DLCs. It is only
compatible with Windows. 

Installation
~~~~~~~~~~~~
The soulsgym package can be installed with pip:

.. code-block:: console

    pip install soulsgym

.. note::
    You need at least Python 3.9 to run soulsgym!

Setup
~~~~~
After installing the package you also need to make some changes to your game settings.

.. warning::
    **If you fail to follow these steps you will most likely get banned from the multiplayer.**

Ban protection
^^^^^^^^^^^^^^
First you need to set your game to offline mode. From the game's main menu go into System -> Network
and set the launch setting to **Play Offline**. Reload the game if necessary. Then create a new
save game. Exclusively use this save game with SoulsGym! Before you go back online, make sure to
delete this game save!

.. _game_settings:

Game settings
^^^^^^^^^^^^^
You have to start your new ``soulsgym`` gamesave as the knight class. Please do not alter the character in any way after loading into the game.
All required character modifications will be performed by the gym.

Key settings
^^^^^^^^^^^^
SoulsGym uses keyboard presses to control the player ingame (see :ref:`core.game_input`). Before launching
an environment you therefore need to set your keys to the internal soulsgym key mapping. Go into System -> Input device -> Key Bindings 
and set your keys to these values:

.. list-table:: Key Bindings
   :widths: 30 30 30
   :header-rows: 1

   * - Type
     - Keyboard
     - Mouse
   * - Run (forward)
     - W
     - 
   * - Run (backward)
     - S
     -
   * - Run (left)
     - A
     -
   * - Run (right)
     - D
     -
   * - Dash/Backstep/Roll
     - Space
     -
   * - Jump
     - Space
     -
   * - Tilt camera up
     - J
     - 
   * - Tilt camera down
     - K
     - 
   * - Tilt camera left
     - O
     - 
   * - Tilt camera right
     - P
     - 
   * - Camera reset / Lock-on
     - Q
     -
   * - Attack (right hand)
     - L
     - ---
   * - Strong attack (right hand)
     - Z
     - 
   * - Attack (left hand)
     - ---
     -
   * - Strong attack (left hand)
     - M
     - 
   * - Use item
     -
     - 
   * - Interact
     - E
     - 
   * - Two-hand weapon
     -
     - 

.. note::
    Fields with --- have to be explicitly unset, empty fields can be whatever.

Graphic settings
^^^^^^^^^^^^^^^^
The game has to be in Windowed mode and profits from stable frame rates. We therefore recommend setting
the graphic settings to

.. list-table:: Graphic Settings
   :widths: 30 30
   :header-rows: 1

   * - Type
     - Setting
   * - Screenmode
     - Windowed (mandatory)
   * - Resolution
     - 800x450 (optional)
