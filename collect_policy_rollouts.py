"""
Iterative World-Models refinement step: deploy the CURRENT trained V+M+C loop
in the real env and record its rollouts (obs/actions/rewards/dones), exactly
like collect_data.py's heuristic rollouts. Mixing these into the dataset lets
M learn to model the states the LEARNED controller actually visits under real
deployment -- not just the states a scripted heuristic visits -- which is
what the original World Models paper does for tasks where a single
dream-only training pass leaves a sim-to-real gap.
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from dino_env import HeadlessDinoEnv
from training_vae import ConvVAE
from train_mdn_rnn import MDNRNN
from train_controller import Controller


def collect_policy_rollouts(num_episodes=400, save_dir="data", start_idx=1900,
                             latent_dim=32, hidden_dim=256, max_steps=1200):
    os.makedirs(save_dir, exist_ok=True)

    vae = ConvVAE(latent_dim=latent_dim).to("cpu")
    vae.load_state_dict(torch.load("checkpoints/vae_final.pth", map_location="cpu"))
    vae.eval()

    m_model = MDNRNN(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim).to("cpu")
    m_model.load_state_dict(torch.load("checkpoints/mdnrnn_final.pth", map_location="cpu"))
    m_model.eval()

    controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=3)
    controller.set_weights(np.load("checkpoints/controller_best.npy"))

    env = HeadlessDinoEnv()

    print(f"Collecting {num_episodes} real-deployment rollouts with the current V+M+C policy...")
    for i in tqdm(range(num_episodes)):
        obs_list, action_list, reward_list, done_list = [], [], [], []
        obs = env.reset()
        done = False
        h_tensor = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

        # Small epsilon of random exploration so the collected data still
        # covers a bit more than exactly the current (imperfect) policy.
        epsilon = 0.1

        while not done:
            frame_tensor = torch.from_numpy(obs).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                mu, _ = vae.encode(frame_tensor)
            z = mu.squeeze(0).numpy()
            h = h_tensor.squeeze().numpy()

            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2])
            else:
                action = controller.act(z, h)

            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[action] = 1.0
            obs_list.append(obs)
            action_list.append(one_hot)

            action_vec = np.zeros((1, 1, 3), dtype=np.float32)
            action_vec[0, 0, action] = 1.0
            z_tensor = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float()
            a_tensor = torch.from_numpy(action_vec)
            with torch.no_grad():
                _, _, _, _, h_tensor = m_model(z_tensor, a_tensor, h_tensor)

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

    print(f"Done. Added {num_episodes} policy rollouts starting at episode_{start_idx:04d}.npz")


if __name__ == "__main__":
    collect_policy_rollouts(num_episodes=400, start_idx=1900)
