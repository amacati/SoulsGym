.. _gym:

SoulsGym in action
==================
After you installed SoulsGym, you are now ready to begin using the environments. 

.. warning::
    Please make sure you followed all the steps in the :ref:`setup <setup>`
    section to prevent bans for cheat detection! Also make sure to delete your game save afterwards!

SoulsGym follows the API of OpenAI's Gym. Here is an implementation of a random agent playing against
Iudex Gundyr:

.. code-block:: python

    import gym
    import soulsgym


    if __name__ == "__main__":
        env = gym.make("SoulsGymIudex-v0")
        state = env.reset()
        done = False
        while not done:
            next_state, reward, done, info = env.step(env.action_space.sample())
        env.close()

It is important to import soulsgym because it can only register the environments to the gym module 
during runtime.

.. note::
    Dark Souls III has to be open before launching the script! Make sure your game fulfills the
    requirements of the environment you are starting (see :ref:`environments <soulsenv>`)!

If we want to get more info about the internal state of the environments we can alter the logging
level:

.. code-block:: python

    import logging

    soulsgym.set_log_level(level=logging.DEBUG)

You should now be able to apply any discrete action Reinforcement Learning algorithm to beat the bosses.