"""The core module of ``soulsgym`` provides all necessary interfaces to the Dark Souls III process.

At the lowest level we directly manipulate the game memory with the :mod:`.memory_manipulator`. The
:mod:`.game_interface` acts as an abstraction layer and allows us to interact with the game as if we
had access to the actual game properties. We cannot however trigger actions directly in game.
Instead we rely on the :mod:`.game_input` to trigger keystrokes in order to control the player.

The :mod:`.logger` module further abstracts the game interface and logs snapshots of the game into a
:class:`.GameState`. This ``GameState`` contains sufficient information about the boss fight to
fulfill the Markov property.

Note:
    For snapshot limitations see :class:`.Logger`.

The :mod:`.static` module offers several game related constants, lists and settings as dictionaries.
"""
