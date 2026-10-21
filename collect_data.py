import os
import numpy as np
from tqdm import tqdm
from dino_env import HeadlessDinoEnv


def heuristic_policy(info, epsilon=0.20):
    """
    Mostly-optimal scripted policy so the dataset contains successful avoidance
    of every obstacle type, plus enough randomness to also see near-misses,
    early/late reactions, and outright failures.

    Obstacle handling:
      - cactus     -> jump when close
      - low_bird   -> duck when close (jumping or running into it kills you)
      - high_bird  -> do nothing (running under is safe; jumping risks a hit)
    """
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2])  # Explore

    dist = info["dist"]
    obs_type = info["type"]
    y_top = info["y_top"]

    if obs_type is None or dist > 80 or dist < 0:
        return 0  # Run

    if obs_type == "cactus":
        return 1  # Jump
    if obs_type == "bird":
        # low_bird has y_top offset 45 -> spawns higher on screen (smaller y value)
        # than high_bird's y_top offset 70. Distinguish by absolute y position.
        # low_bird: y_top = ground_y - 45 ; high_bird: y_top = ground_y - 70
        # Since ground_y=120: low_bird y_top=75, high_bird y_top=50
        if y_top >= 65:  # low bird (closer to ground) -> duck
            return 2
        else:  # high bird -> do nothing
            return 0
    return 0


def collect_rollouts(num_episodes=1500, save_dir="data", max_steps=1200):
    os.makedirs(save_dir, exist_ok=True)
    env = HeadlessDinoEnv()

    print(f"Starting data collection: {num_episodes} episodes...")

    for ep in tqdm(range(num_episodes)):
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []

        obs = env.reset()
        done = False
        info = {"dist": 9999, "type": None, "y_top": None, "y_bottom": None}

        # Vary exploration rate per-episode so the dataset has a spread of
        # "skilled" and "clumsy" trajectories (skilled -> longer episodes with
        # more obstacle variety, clumsy -> more failure/collision examples).
        epsilon = np.random.choice([0.05, 0.15, 0.25, 0.35])

        while not done:
            action = heuristic_policy(info, epsilon=epsilon)

            one_hot_action = np.zeros(3, dtype=np.float32)
            one_hot_action[action] = 1.0

            obs_list.append(obs)
            action_list.append(one_hot_action)

            obs, reward, done, step_info = env.step(action)
            info = step_info["next_obstacle"]

            reward_list.append(reward)
            done_list.append(done)

            if len(obs_list) >= max_steps:
                break

        np.savez_compressed(
            os.path.join(save_dir, f"episode_{ep:04d}.npz"),
            obs=np.array(obs_list, dtype=np.uint8),         # Shape: (T, 1, 64, 64)
            actions=np.array(action_list, dtype=np.float32),# Shape: (T, 3)
            rewards=np.array(reward_list, dtype=np.float32),# Shape: (T,)
            dones=np.array(done_list, dtype=np.bool_)        # Shape: (T,)
        )

    print(f"Collection complete! Saved {num_episodes} rollouts in '{save_dir}/'.")


if __name__ == "__main__":
    collect_rollouts(num_episodes=1500)
