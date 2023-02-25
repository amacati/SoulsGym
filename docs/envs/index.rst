soulsgym.envs
=============

.. automodule:: soulsgym.envs

.. toctree::

    soulsgym
    iudex_env

Game speed
~~~~~~~~~~

The :ref:`core.speedhack` module allows us to suspend the game loop after a single step. It also
enables us to speed up the game during environment steps. Although this is desirable to accelerate
training, please note that we cannot arbitrarily accelerate the game. The speed hack only changes
the time calls to the OS and trigger a game loop step. If the game speed exceeds the performance of
the training hardware, the loop won't be able to accelerate further.

.. warning::
    Do not increase the game speed to values your hardware can't handle! Remember that the game has
    to render images at the same accelerated pace. We recommend to limit values to the interval of
    [1, 3].