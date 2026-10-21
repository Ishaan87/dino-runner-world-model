import os
import time
import pygame
import numpy as np
import torch
from dino_env import HeadlessDinoEnv
from training_vae import ConvVAE
from train_mdn_rnn import MDNRNN
from train_controller import Controller

ACTION_NAMES = {0: "RUN", 1: "JUMP", 2: "DUCK"}


def run_live_test(num_episodes=5, latent_dim=32, hidden_dim=256):
    pygame.init()
    screen_w, screen_h = 600, 150
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("World Model Agent - Live Test")
    clock = pygame.time.Clock()

    # Load Vision Model (V)
    vae = ConvVAE(latent_dim=latent_dim).to("cpu")
    vae.load_state_dict(torch.load("checkpoints/vae_final.pth", map_location="cpu"))
    vae.eval()

    # Load Memory Model (M)
    m_model = MDNRNN(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim).to("cpu")
    m_model.load_state_dict(torch.load("checkpoints/mdnrnn_final.pth", map_location="cpu"))
    m_model.eval()

    # Load Controller (C) -- same 3-action controller trained in dream
    controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=3)
    controller.set_weights(np.load("checkpoints/controller_best.npy"))

    env = HeadlessDinoEnv(width=screen_w, height=screen_h)

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_score = 0
        # h starts at zero -- identical to how dream rollouts start in
        # train_controller.py. No privileged warm-up here or there.
        h_tensor = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

        print(f"\n--- Starting Episode {ep} ---")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 1. Vision (V): Encode frame
            frame_tensor = torch.from_numpy(obs).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                mu, _ = vae.encode(frame_tensor)
                z = mu.squeeze(0).numpy()

            # 2. Controller (C): pick run / jump / duck
            h = h_tensor.squeeze().numpy()
            action = controller.act(z, h)

            # 3. Memory (M): advance hidden state with the action actually taken
            action_vec = np.zeros((1, 1, 3), dtype=np.float32)
            action_vec[0, 0, action] = 1.0
            z_tensor = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float()
            a_tensor = torch.from_numpy(action_vec)

            with torch.no_grad():
                _, _, _, _, h_tensor = m_model(z_tensor, a_tensor, h_tensor)

            # 4. Environment Step
            obs, reward, done, _ = env.step(action)
            total_score += 1

            # 5. Live Display
            screen.blit(env.surface, (0, 0))
            font = pygame.font.SysFont("Arial", 16)
            txt = font.render(f"Score: {total_score} | Action: {ACTION_NAMES[action]}", True, (255, 0, 0))
            screen.blit(txt, (10, 10))

            pygame.display.flip()
            clock.tick(30)

        print(f"Episode {ep} Finished | Real Score: {total_score}")
        time.sleep(1.0)

    pygame.quit()


if __name__ == "__main__":
    run_live_test(num_episodes=5)
