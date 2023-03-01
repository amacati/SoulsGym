import logging

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
    attacks = {}
    try:
        while True:
            state = env.reset()
            done = False
            while not done:
                next_state, reward, done, info = env.step(env.action_space.sample())
                if next_state.boss_animation != state.boss_animation:
                    if "Attack" in state.boss_animation:
                        if state.boss_animation not in attacks:
                            attacks[state.boss_animation] = [state.boss_animation_duration]
                        else:
                            attacks[state.boss_animation].append(state.boss_animation_duration)
                    print(state.boss_animation, f"{state.boss_animation_duration:.2f}")
                state = next_state
    finally:
        print("Iudex" if next_state.boss_hp == 0 else "Player", " defeated.")
        print(f"Env speed: {env.game_speed}")
        env.close()
        results = sorted([(key, val) for key, val in attacks.items() if val])
        for key, val in results:
            print(f"{key}: {sum(val) / len(val) :.2f} ({len(val)} samples)")
