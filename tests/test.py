import logging
import time
import datetime

import gym
import soulsgym
from soulsgym.core.game_input import GameInput  # noqa: F401

if __name__ == "__main__":
    logging.basicConfig(filename="soulsgym.log",
                        filemode="w",
                        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    soulsgym.set_log_level(level=logging.INFO)
    env = gym.make("SoulsGymIudex-v0")
    try:
        state = env.reset()
        done = False
        while not done:
            next_state, reward, done, info = env.step(19)
    finally:
        print("Iudex" if next_state.boss_hp == 0 else "Player", " defeated.")
        env.close()
