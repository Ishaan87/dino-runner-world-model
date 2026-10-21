import numpy as np
import torch
from dino_env import HeadlessDinoEnv
from training_vae import ConvVAE
from train_mdn_rnn import MDNRNN
from train_controller import Controller

ACTION_NAMES = {0: "RUN", 1: "JUMP", 2: "DUCK"}
latent_dim, hidden_dim = 32, 256

vae = ConvVAE(latent_dim=latent_dim).to("cpu")
vae.load_state_dict(torch.load("checkpoints/vae_final.pth", map_location="cpu"))
vae.eval()

m_model = MDNRNN(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim).to("cpu")
m_model.load_state_dict(torch.load("checkpoints/mdnrnn_final.pth", map_location="cpu"))
m_model.eval()

controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=3)
controller.set_weights(np.load("checkpoints/controller_best.npy"))

for run in range(5):
    env = HeadlessDinoEnv()
    obs = env.reset()
    done = False
    h_tensor = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
    steps = 0
    printed = False
    while not done and steps < 250:
        frame_tensor = torch.from_numpy(obs).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            mu, _ = vae.encode(frame_tensor)
            z = mu.squeeze(0).numpy()
        h = h_tensor.squeeze().numpy()
        action = controller.act(z, h)

        info_before = env._get_next_obstacle_info()
        if info_before["dist"] < 90 and not printed:
            print(f"run {run} step {steps}: obstacle={info_before['type']} dist={info_before['dist']} "
                  f"y_top={info_before['y_top']} -> action={ACTION_NAMES[action]}")

        action_vec = np.zeros((1, 1, 3), dtype=np.float32)
        action_vec[0, 0, action] = 1.0
        z_tensor = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float()
        a_tensor = torch.from_numpy(action_vec)
        with torch.no_grad():
            _, _, _, _, h_tensor = m_model(z_tensor, a_tensor, h_tensor)

        obs, reward, done, _ = env.step(action)
        steps += 1
        if done:
            print(f"run {run}: DIED at step {steps}")
            printed = True
