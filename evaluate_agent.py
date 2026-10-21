import numpy as np
import torch
from dino_env import HeadlessDinoEnv
from training_vae import ConvVAE
from train_mdn_rnn import MDNRNN
from train_controller import Controller


def evaluate(num_episodes=50, latent_dim=32, hidden_dim=256, max_steps=3000, verbose=True):
    vae = ConvVAE(latent_dim=latent_dim).to("cpu")
    vae.load_state_dict(torch.load("checkpoints/vae_final.pth", map_location="cpu"))
    vae.eval()

    m_model = MDNRNN(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim).to("cpu")
    m_model.load_state_dict(torch.load("checkpoints/mdnrnn_final.pth", map_location="cpu"))
    m_model.eval()

    controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=3)
    controller.set_weights(np.load("checkpoints/controller_best.npy"))

    env = HeadlessDinoEnv()
    scores = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        h_tensor = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

        while not done and steps < max_steps:
            frame_tensor = torch.from_numpy(obs).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                mu, _ = vae.encode(frame_tensor)
                z = mu.squeeze(0).numpy()

            h = h_tensor.squeeze().numpy()
            action = controller.act(z, h)

            action_vec = np.zeros((1, 1, 3), dtype=np.float32)
            action_vec[0, 0, action] = 1.0
            z_tensor = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float()
            a_tensor = torch.from_numpy(action_vec)

            with torch.no_grad():
                _, _, _, _, h_tensor = m_model(z_tensor, a_tensor, h_tensor)

            obs, reward, done, _ = env.step(action)
            steps += 1

        scores.append(steps)
        if verbose:
            print(f"Episode {ep + 1:03d}: {steps} steps")

    scores = np.array(scores)
    print(f"\nOver {num_episodes} episodes -> mean={scores.mean():.1f}  "
          f"median={np.median(scores):.1f}  min={scores.min()}  max={scores.max()}")
    return scores


if __name__ == "__main__":
    evaluate(num_episodes=50)
