import os
import glob
import numpy as np
import torch
import cma
from tqdm import tqdm
from train_mdn_rnn import MDNRNN


class Controller:
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        self.input_dim = latent_dim + hidden_dim  # 288
        self.action_dim = action_dim              # 3: run / jump / duck
        self.num_params = (self.input_dim * self.action_dim) + self.action_dim

    def set_weights(self, flat_params):
        w_size = self.input_dim * self.action_dim
        self.W = flat_params[:w_size].reshape(self.action_dim, self.input_dim)
        self.b = flat_params[w_size:]

    def act(self, z, h):
        state = np.concatenate([z, h], axis=-1)
        logits = np.dot(self.W, state) + self.b
        return int(np.argmax(logits))


def rollout_in_dream(controller, m_model, episodes, max_dream_steps=800, temperature=1.15):
    """
    Dream rollout that mirrors real deployment exactly:
    the episode starts from a real recorded first frame (z0) with a
    zero-initialized RNN hidden state (h0) -- there is no privileged
    "warm-up" using future real frames, because that information does
    not exist at deployment time in test_live.py.
    """
    ep = episodes[np.random.randint(0, len(episodes))]

    h = np.zeros(m_model.hidden_dim, dtype=np.float32)
    z = ep["latents"][0].copy()

    dream_steps = 0
    jump_count = 0
    duck_count = 0

    with torch.no_grad():
        for step in range(max_dream_steps):
            action_idx = controller.act(z, h)
            if action_idx == 1:
                jump_count += 1
            elif action_idx == 2:
                duck_count += 1

            action_vec = np.zeros((1, 1, 3), dtype=np.float32)
            action_vec[0, 0, action_idx] = 1.0

            z_tensor = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float()
            a_tensor = torch.from_numpy(action_vec)
            h_tensor = torch.from_numpy(h).unsqueeze(0).unsqueeze(0).float()

            pi, mu, sigma, done_pred, h_next = m_model(z_tensor, a_tensor, h_tensor)

            if done_pred.squeeze().item() > 0.55:
                break

            pi_np = pi.squeeze().numpy()
            mu_np = mu.squeeze(0).squeeze(0).numpy()
            sigma_np = sigma.squeeze(0).squeeze(0).numpy()

            pi_np = np.clip(pi_np, 1e-6, 1.0)
            pi_np = pi_np / np.sum(pi_np)

            k = np.random.choice(len(pi_np), p=pi_np)
            eps = np.random.randn(*mu_np[k].shape)
            z = mu_np[k] + (sigma_np[k] * np.sqrt(temperature)) * eps
            h = h_next.squeeze().numpy()

            dream_steps += 1

    # Fitness: steps survived, penalized both for spamming jump/duck AND for
    # never using them at all. The latter matters a lot here: M's done-head
    # is only reliable within roughly the horizon it was trained on (a few
    # hundred steps); push a rollout far beyond that and the hidden state
    # drifts off-distribution and the done-head can flatline near zero no
    # matter what happens, so a policy that always says "run" can rack up
    # a huge score purely by exploiting that blind spot without ever
    # learning to actually react to obstacles. A real episode encounters an
    # obstacle roughly every 45-90 steps, so a policy that goes hundreds of
    # dream-steps without ever jumping or ducking is almost certainly
    # gaming the model rather than surviving it.
    jump_ratio = jump_count / max(1, dream_steps)
    duck_ratio = duck_count / max(1, dream_steps)
    penalty = 0.0
    if jump_ratio > 0.30:
        penalty += (jump_ratio - 0.30) * 400.0
    if duck_ratio > 0.30:
        penalty += (duck_ratio - 0.30) * 400.0
    if dream_steps > 40:
        expected_min_jump = dream_steps / 300.0   # ~1 jump per 300 steps, generous floor
        expected_min_duck = dream_steps / 400.0
        if jump_count < expected_min_jump:
            penalty += (expected_min_jump - jump_count) * 40.0
        if duck_count < expected_min_duck:
            penalty += (expected_min_duck - duck_count) * 40.0

    fitness = max(0.0, float(dream_steps) - penalty)
    return fitness


def train_controller_in_dream(generations=60, pop_size=32, latent_dim=32, hidden_dim=256):
    print("Training Controller (Run / Jump / Duck) in dreams...")

    m_model = MDNRNN(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim).to("cpu")
    m_model.load_state_dict(torch.load("checkpoints/mdnrnn_final.pth", map_location="cpu"))
    m_model.eval()

    files = sorted(glob.glob("latent_data/*.npz"))
    episodes = []
    for f in files:
        data = np.load(f)
        if len(data["latents"]) < 2:
            continue
        episodes.append({
            "latents": data["latents"],
            "actions": data["actions"]
        })
    print(f"Loaded {len(episodes)} latent episodes for dream rollouts.")

    controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=3)
    init_params = np.random.randn(controller.num_params) * 0.1
    es = cma.CMAEvolutionStrategy(init_params, 0.2, {"popsize": pop_size})

    history = {"gen": [], "best": [], "mean": []}
    best_ever = -1
    for gen in range(1, generations + 1):
        candidates = es.ask()
        fitness_scores = []

        for flat_weights in tqdm(candidates, desc=f"Gen {gen:02d}/{generations}", leave=False):
            controller.set_weights(flat_weights)
            scores = [
                # Capped well below M's uncharted long-horizon drift zone --
                # see the comment in rollout_in_dream's fitness computation.
                rollout_in_dream(controller, m_model, episodes, max_dream_steps=300, temperature=0.9)
                for _ in range(6)
            ]
            fitness_scores.append(np.mean(scores))

        es.tell(candidates, [-score for score in fitness_scores])

        best_score = max(fitness_scores)
        mean_score = np.mean(fitness_scores)
        print(f"Gen {gen:02d}/{generations} | Best Fitness: {best_score:.1f} | Avg: {mean_score:.1f}")

        history["gen"].append(gen)
        history["best"].append(best_score)
        history["mean"].append(mean_score)
        os.makedirs("checkpoints", exist_ok=True)
        np.savez("checkpoints/controller_training_history.npz", **{k: np.array(v) for k, v in history.items()})

        if best_score > best_ever:
            best_ever = best_score
            best_idx = int(np.argmax(fitness_scores))
            os.makedirs("checkpoints", exist_ok=True)
            np.save("checkpoints/controller_best.npy", candidates[best_idx])

    print(f"Saved best controller weights (fitness={best_ever:.1f}) to 'checkpoints/controller_best.npy'.")


if __name__ == "__main__":
    train_controller_in_dream(generations=60, pop_size=32)
