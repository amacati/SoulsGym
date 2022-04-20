.. SoulsGym documentation master file, created by
   sphinx-quickstart on Tue Apr 19 18:27:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SoulsGym documentation
====================================

SoulsGym is an extension of the OpenAI Gym module for Reinforcement Learning in Python.
With SoulsGym you can train Reinforcement Learning algorithms on Dark Souls III bosses using the Gym API.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/setup
   getting_started/gym

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Python API

   soulsgym
   envs
   envs.soulsgym
   envs.iudex_env
   core
   core.game_input
   core.game_interface
   core.game_state
   core.game_window
   core.logger
   core.memory_manipulator
   core.static
   core.utils
   exception

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/stability
   notes/acknowledgements


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
