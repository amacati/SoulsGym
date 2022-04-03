import logging

import gym
import soulsgym  # noqa: F401

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make("SoulsGymIudex-v0")
    try:
        for _ in range(3):
            state = env.reset()
            done = False
            while not done:
                next_state, reward, done, info = env.step(env.action_space.sample())
    finally:
        env.close()
