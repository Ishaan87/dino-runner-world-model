import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ==========================================
# 1. Latent Sequence Dataset Loader
# ==========================================
class LatentSequenceDataset(Dataset):
    def __init__(self, latent_dir="latent_data", seq_len=100):
        self.files = sorted(glob.glob(os.path.join(latent_dir, "*.npz")))
        self.seq_len = seq_len
        self.episodes = []

        print(f"Loading {len(self.files)} latent episodes into memory...")
        for f in tqdm(self.files):
            d = np.load(f)
            self.episodes.append({
                "latents": d["latents"],  # (T, 32)
                "actions": d["actions"],  # (T, 3)
                "dones": d["dones"]       # (T,)
            })

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        T = len(ep["latents"])
        seq_len = min(self.seq_len, T)

        if T <= seq_len:
            start = 0
        else:
            roll = np.random.rand()
            if roll < 0.4:
                # Force-include the episode's final transition, where dones=True.
                # A pure random window (as in the original implementation) can
                # never reach the last index, so the collision-prediction head
                # would never see a single true positive across the whole dataset.
                start = T - seq_len
            elif roll < 0.7:
                # Force-include the true episode start (h=0 paired with the
                # canonical "just reset, nothing nearby" frame) -- this is
                # exactly the state real deployment begins from every episode,
                # and pure random starts rarely land on index 0 for long T.
                start = 0
            else:
                start = np.random.randint(0, T - seq_len + 1)

        end = start + seq_len  # exclusive; end == T means window reaches the true last frame
        latents = ep["latents"][start:end]     # z_t,  length seq_len
        actions = ep["actions"][start:end]     # a_t,  length seq_len
        dones = ep["dones"][start:end]         # done_t, length seq_len (True only possibly at the last row)

        if end < T:
            next_latents = ep["latents"][start + 1 : end + 1]  # z_(t+1), fully valid
            z_valid_len = seq_len
        else:
            # No recorded frame exists after a terminal collision -- pad with a
            # self-loop copy and mask it out of the z-prediction (NLL) loss via
            # z_valid_len; the done/BCE target at that final row is still real.
            next_latents = np.concatenate([ep["latents"][start + 1 : end], latents[-1:]], axis=0)
            z_valid_len = seq_len - 1

        return (
            torch.tensor(latents, dtype=torch.float32),        # z_t
            torch.tensor(actions, dtype=torch.float32),         # a_t
            torch.tensor(next_latents, dtype=torch.float32),    # z_(t+1) target
            torch.tensor(dones, dtype=torch.float32),           # done_t target
            seq_len,
            z_valid_len,
        )

def pad_collate(batch):
    z_t, a_t, next_z, dones, seq_lens, z_valid_lens = zip(*batch)
    z_t_pad = pad_sequence(z_t, batch_first=True, padding_value=0.0)
    a_t_pad = pad_sequence(a_t, batch_first=True, padding_value=0.0)
    next_z_pad = pad_sequence(next_z, batch_first=True, padding_value=0.0)
    dones_pad = pad_sequence(dones, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(seq_lens, dtype=torch.long)          # for the done/BCE mask
    z_lengths = torch.tensor(z_valid_lens, dtype=torch.long)    # for the z-prediction/NLL mask
    return z_t_pad, a_t_pad, next_z_pad, dones_pad, lengths, z_lengths


# ==========================================
# 2. MDN-RNN Architecture (M Model)
# ==========================================
class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.num_gaussians = num_gaussians
        self.hidden_dim = hidden_dim

        # Input: [z_t, a_t]
        self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)

        # Output heads for MDN parameters:
        # pi: (num_gaussians)
        # mu: (num_gaussians * latent_dim)
        # sigma: (num_gaussians * latent_dim)
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

        # Done prediction head (collision)
        self.fc_done = nn.Linear(hidden_dim, 1)

    def forward(self, z, a, h=None):
        # Concatenate current state and action
        x = torch.cat([z, a], dim=-1)
        out, h_next = self.rnn(x, h)

        batch_size, seq_len, _ = out.shape

        pi = F.softmax(self.fc_pi(out), dim=-1) # (B, T, K)
        mu = self.fc_mu(out).view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        sigma = torch.exp(self.fc_sigma(out)).view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        done = torch.sigmoid(self.fc_done(out)).squeeze(-1) # (B, T)

        return pi, mu, sigma, done, h_next


# ==========================================
# 3. MDN Loss Function (Log-Likelihood)
# ==========================================
def mdn_loss_fn(next_z, dones, pi, mu, sigma, done_preds, lengths, z_lengths,
                done_pos_weight=80.0, done_loss_scale=50.0):
    # Expand target z to match Gaussian dimensions: (B, T, K, Latent)
    target = next_z.unsqueeze(2).expand_as(mu)

    # Compute probability density for each Gaussian component: N(target | mu, sigma)
    var = sigma ** 2
    log_gaussian = -0.5 * (((target - mu) ** 2) / (var + 1e-7)) - torch.log(sigma + 1e-7) - 0.5 * math.log(2 * math.pi)
    log_gaussian = torch.sum(log_gaussian, dim=-1) # Sum over latent dimensions -> (B, T, K)

    # Log-sum-exp over mixtures: log(sum(pi * N(...)))
    log_pi = torch.log(pi + 1e-7)
    log_prob = torch.logsumexp(log_pi + log_gaussian, dim=-1) # (B, T)

    # z-prediction is only valid where a real "next frame" was recorded --
    # the row for a terminal (done=True) transition has no such frame.
    z_mask = torch.arange(next_z.size(1), device=next_z.device)[None, :] < z_lengths[:, None]
    # done/collision target is valid for every real (non-padding) row, including the terminal one.
    done_mask = torch.arange(next_z.size(1), device=next_z.device)[None, :] < lengths[:, None]

    nll_loss = -torch.mean(log_prob[z_mask])

    bce_per_elem = F.binary_cross_entropy(done_preds[done_mask], dones[done_mask], reduction="none")
    pos_weight = torch.where(dones[done_mask] > 0.5,
                              torch.full_like(bce_per_elem, done_pos_weight),
                              torch.ones_like(bce_per_elem))
    # Weighted AVERAGE (normalize by total weight, not element count) so the
    # rare positive rows actually move the loss instead of being drowned out
    # by the ~99% negative rows in every 100-step window.
    bce_loss = (bce_per_elem * pos_weight).sum() / pos_weight.sum()

    # The NLL term for a 32-d continuous mixture routinely sits around -50,
    # which would otherwise swamp the ~0.3-0.7 BCE term and starve the
    # done-head of gradient. Scale BCE up so both heads actually get trained.
    return nll_loss + done_loss_scale * bce_loss, nll_loss, bce_loss


# ==========================================
# 4. Training Loop
# ==========================================
def train_mdnrnn(epochs=20, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MDN-RNN on: {device}")

    dataset = LatentSequenceDataset(latent_dir="latent_data", seq_len=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "loss": [], "nll": [], "bce": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_nll, total_bce = 0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for z_t, a_t, next_z, dones, lengths, z_lengths in pbar:
            z_t, a_t = z_t.to(device), a_t.to(device)
            next_z, dones = next_z.to(device), dones.to(device)
            lengths, z_lengths = lengths.to(device), z_lengths.to(device)

            optimizer.zero_grad()
            pi, mu, sigma, done_preds, _ = model(z_t, a_t)

            loss, nll, bce = mdn_loss_fn(next_z, dones, pi, mu, sigma, done_preds, lengths, z_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_bce += bce.item()

            pbar.set_postfix({"Loss": f"{loss.item():.3f}", "NLL": f"{nll.item():.3f}", "BCE": f"{bce.item():.3f}"})

        n = len(dataloader)
        print(f"Epoch {epoch} Complete | Avg Loss: {total_loss / n:.4f}")

        history["epoch"].append(epoch)
        history["loss"].append(total_loss / n)
        history["nll"].append(total_nll / n)
        history["bce"].append(total_bce / n)
        os.makedirs("checkpoints", exist_ok=True)
        np.savez("checkpoints/mdnrnn_training_history.npz", **{k: np.array(v) for k, v in history.items()})

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mdnrnn_final.pth")
    print("M Model Training Complete! Saved 'checkpoints/mdnrnn_final.pth'.")

if __name__ == "__main__":
    train_mdnrnn(epochs=20, batch_size=64, lr=1e-3)