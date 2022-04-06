import logging

import gym
import json
import soulsgym  # noqa: F401

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make("SoulsGymIudex-v0")
    try:
        for i in range(500):
            print(f"Starting episode {i+1}")
            state = env.reset()
            states = []
            states.append(state)
            done = False
            while not done:
                next_state, reward, done, info = env.step(env.action_space.sample())
                states.append(next_state)
    finally:
        env.close()
        with open("unknown_animations.json", "w") as f:
            anim_dict = {
                "boss": env.unknown_boss_animations,
                "player": env.unknown_player_animations
            }
            json.dump(anim_dict, f)
        print(states)
