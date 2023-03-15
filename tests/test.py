import logging

import gymnasium as gym
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
    env = gym.make("SoulsGymIudex-v0", init_pose_randomization=True, game_speed=3)
    try:
        while True:
            obs, info = env.reset()
            terminated = False
            while not terminated:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
    finally:
        print("Iudex" if next_obs["boss_hp"] == 0 else "Player", " defeated.")
        env.close()
