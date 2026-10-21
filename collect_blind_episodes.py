"""
Adds dedicated 'blind' episodes to data/ where the agent always RUNs and
never reacts to obstacles at all. The scripted heuristic in collect_data.py
reacts correctly most of the time and only fails via occasional epsilon-
random noise, so M rarely if ever saw a long, systematic "ignored every
obstacle" trajectory during training. Without that signal M can't reliably
learn "no reaction near an obstacle -> collision", which is exactly the
blind spot a CMA-ES controller will find and exploit in dream rollouts.
These episodes supply that missing causal signal directly.
"""
import os
import numpy as np
from tqdm import tqdm
from dino_env import HeadlessDinoEnv


def collect_blind_rollouts(num_episodes=400, save_dir="data", start_idx=1500, max_steps=1200):
    os.makedirs(save_dir, exist_ok=True)
    env = HeadlessDinoEnv()

    print(f"Collecting {num_episodes} blind (always-RUN) episodes...")
    for i in tqdm(range(num_episodes)):
        obs_list, action_list, reward_list, done_list = [], [], [], []
        obs = env.reset()
        done = False

        while not done:
            action = 0  # always RUN -- deliberately ignores every obstacle
            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[action] = 1.0

            obs_list.append(obs)
            action_list.append(one_hot)

            obs, reward, done, _ = env.step(action)
            reward_list.append(reward)
            done_list.append(done)

            if len(obs_list) >= max_steps:
                break

        ep_id = start_idx + i
        np.savez_compressed(
            os.path.join(save_dir, f"episode_{ep_id:04d}.npz"),
            obs=np.array(obs_list, dtype=np.uint8),
            actions=np.array(action_list, dtype=np.float32),
            rewards=np.array(reward_list, dtype=np.float32),
            dones=np.array(done_list, dtype=np.bool_),
        )

    print(f"Done. Added {num_episodes} blind episodes starting at episode_{start_idx:04d}.npz")


if __name__ == "__main__":
    collect_blind_rollouts(num_episodes=400, start_idx=1500)
