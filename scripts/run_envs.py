import fire
import gymnasium
import datetime

import soulsgym  # noqa: F401


def main(env: str = "SoulsGymIudex-v0"):
    env = gymnasium.make(env, game_speed=3., random_player_pose=True)
    try:
        start = datetime.datetime.now()
        while True:
            env.reset()
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
            print("Current runtime: " + str(datetime.datetime.now() - start).split('.')[0])
    finally:
        env.close()


if __name__ == "__main__":
    fire.Fire(main)
