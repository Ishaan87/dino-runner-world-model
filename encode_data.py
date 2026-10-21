import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from training_vae import ConvVAE  # Imports your Phase 2 architecture

def encode_all_episodes(data_dir="data", save_dir="latent_data", checkpoint_path="checkpoints/vae_final.pth", latent_dim=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load trained VAE encoder and freeze weights
    vae = ConvVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vae.eval()

    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Encoding {len(files)} episodes into latent vectors on {device}...")

    with torch.no_grad():
        for file_path in tqdm(files):
            raw_data = np.load(file_path)
            obs = raw_data["obs"]        # Shape: (T, 1, 64, 64)
            actions = raw_data["actions"]# Shape: (T, 3)
            rewards = raw_data["rewards"]# Shape: (T,)
            dones = raw_data["dones"]    # Shape: (T,)

            # Batch encode frames
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device) / 255.0
            mu, logvar = vae.encode(obs_tensor)
            
            # Use deterministic mean mu as the latent state representation
            latents = mu.cpu().numpy() # Shape: (T, 32)

            base_name = os.path.basename(file_path)
            np.savez_compressed(
                os.path.join(save_dir, base_name),
                latents=latents,
                actions=actions,
                rewards=rewards,
                dones=dones
            )

    print(f"Encoding complete! Saved latent files to '{save_dir}/'.")

if __name__ == "__main__":
    encode_all_episodes()