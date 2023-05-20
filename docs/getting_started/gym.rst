.. _gym:

SoulsGym in action
==================
After you installed SoulsGym, you are now ready to begin using the environments. 

.. warning::
    Please make sure you followed all the steps in the :ref:`setup <setup>`
    section to prevent bans for cheat detection! Also make sure to delete your game save afterwards!

SoulsGym follows the API of Farama's gymnasium. Here is an implementation of a random agent playing against
Iudex Gundyr:

.. code-block:: python

    import gymnasium
    import soulsgym


    if __name__ == "__main__":
        env = gymnasium.make("SoulsGymIudex-v0")
        obs, info = env.reset()
        terminated = False
        while not terminated:
            next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.close()

It is important to import soulsgym because it can only register the environments to the gymnasium module 
during runtime.

.. note::
    Dark Souls III has to be open before launching the script! Make sure your game save fulfills the
    requirements (see :ref:`game settings <game_settings>`)!

If we want to get more info about the internal state of the environments we can alter the logging
level:

.. code-block:: python

    import logging

    soulsgym.set_log_level(level=logging.DEBUG)

You should now be able to apply any discrete action Reinforcement Learning algorithm to beat the bosses.