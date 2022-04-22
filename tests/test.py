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
                        level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler())
    soulsgym.set_log_level(level=logging.DEBUG)
    env = gym.make("SoulsGymIudex-v0")
    t_start = time.time()
    sample_count = 0
    try:
        i = 1
        while datetime.timedelta(seconds=time.time() - t_start).days == 0:
            print(f"Starting episode {i}")
            state = env.reset()
            done = False
            while not done:
                next_state, reward, done, info = env.step(env.action_space.sample())
                sample_count += 1
            seconds = round(time.time() - t_start)
            print(f"Current gym uptime: {str(datetime.timedelta(seconds=seconds))}")
            i += 1
    finally:
        t_end = time.time()
        print(f"Total gym run time: {str(datetime.timedelta(seconds=round(t_end-t_start)))}")
        print(f"Total sample count: {sample_count}")
        env.close()
