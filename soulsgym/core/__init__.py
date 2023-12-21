"""The core module of ``soulsgym`` provides all necessary interfaces to the Dark Souls III process.

At the lowest level we directly manipulate the game memory with the :mod:`.memory_manipulator`. This
memory manipulator is leveraged in the :mod:`soulsgym.games` module to provide an abstraction layer
and allows us to interact with the game as if we had access to the actual game properties.

While we can change the game memory, we cannot trigger actions directly in game. Instead, we rely on
the :mod:`.game_input` to trigger keystrokes to control the player.

To pause and accelerate the game, the :mod:`.speedhack` module provides functions to inject a DLL
into the game process and dynamically change the game loop speed.

In addition to the ground truth game states like the exact player coordinates etc., we also read out
the game's current frame with the :mod:`.game_window` module. It features a Python wrapper around a
``window_capture`` submodule for screen capture. The capture itself is implemented in C++ with
DirectX to enable fast and efficient screen capture.

The :mod:`.static` module offers several game related constants, lists and settings as dictionaries.
"""
