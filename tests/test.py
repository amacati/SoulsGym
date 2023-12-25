import logging

import fire
import gymnasium
import soulsgym


def main(boss: str):
    env = gymnasium.make(f"SoulsGym{boss}-v0")
    try:
        for _ in range(3):
            obs, info = env.reset()
            terminated, truncated = False, False
            while not terminated and not truncated:
                next_obs, reward, terminated, truncated, info = env.step(19)
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(filename="soulsgym.log",
                        filemode="w",
                        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    soulsgym.set_log_level(level=logging.INFO)
    fire.Fire(main)
