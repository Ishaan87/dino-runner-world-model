import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# ==========================================
# 1. Dataset Loader for .npz Episode Files
# ==========================================
class DinoRolloutDataset(Dataset):
    def __init__(self, data_dir="data", max_episodes=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_episodes:
            self.files = self.files[:max_episodes]

        # Load all frames into RAM for fast batching
        frames_list = []
        print(f"Loading {len(self.files)} episode files into memory...")
        for f in tqdm(self.files):
            data = np.load(f)
            # data['obs'] shape: (T, 1, 64, 64), uint8
            frames_list.append(data["obs"])

        self.frames = np.concatenate(frames_list, axis=0)
        print(f"Dataset ready. Total frames: {len(self.frames)}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Convert uint8 [0, 255] to float32 [0.0, 1.0]
        frame = self.frames[idx].astype(np.float32) / 255.0
        return torch.tensor(frame, dtype=torch.float32)


# ==========================================
# 2. Convolutional VAE Architecture (V)
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x64x64 -> 256x2x2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> (256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),# -> (256, 2, 2)
            nn.ReLU()
        )

        # Latent heads
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder projection
        self.decoder_input = nn.Linear(latent_dim, 256 * 2 * 2)

        # Decoder: 256x2x2 -> 1x64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1), # -> (256, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # -> (1, 64, 64)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# ==========================================
# 3. Loss Function (MSE + KL Divergence)
# ==========================================
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (per-pixel MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ==========================================
# 4. Training Loop
# ==========================================
def train_vae(epochs=15, batch_size=128, lr=1e-4, latent_dim=32, beta=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("recon_samples", exist_ok=True)

    dataset = DinoRolloutDataset(data_dir="data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "loss": [], "recon": [], "kl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            loss, recon, kl = vae_loss_function(recon_batch, batch, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()

            pbar.set_postfix({
                "Loss": f"{loss.item() / len(batch):.2f}",
                "MSE": f"{recon.item() / len(batch):.2f}",
                "KL": f"{kl.item() / len(batch):.2f}"
            })

        avg_loss = train_loss / len(dataset)
        avg_recon = train_recon / len(dataset)
        avg_kl = train_kl / len(dataset)
        print(f"==> Epoch {epoch} Complete | Avg Loss/Frame: {avg_loss:.2f}")

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        np.savez("checkpoints/vae_training_history.npz", **{k: np.array(v) for k, v in history.items()})

        # Save visual comparison: Top row = Real, Bottom row = Decoded
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader))[:8].to(device)
            recon_sample, _, _ = model(sample_batch)
            comparison = torch.cat([sample_batch, recon_sample], dim=0)
            save_image(comparison.cpu(), f"recon_samples/epoch_{epoch:02d}.png", nrow=8)

        # Save model weights
        torch.save(model.state_dict(), f"checkpoints/vae_epoch_{epoch:02d}.pth")

    # Save final model for Phase 3
    torch.save(model.state_dict(), "checkpoints/vae_final.pth")
    print("Training Complete! Saved 'checkpoints/vae_final.pth'.")


if __name__ == "__main__":
    train_vae(epochs=15, batch_size=128, lr=1e-4, latent_dim=32, beta=1.0)