# Dino runner world model

A **World Models** (Ha & Schmidhuber) agent that learns to play Chrome's offline dinosaur game.

## Architecture

- **V (vision):** a convolutional VAE compresses each frame into a 32-d latent (`training_vae.py`).
- **M (memory):** an MDN-RNN predicts the next latent from the current latent and action (`train_mdn_rnn.py`).
- **C (controller):** a small linear controller on `[z, h]` (288 inputs, 3 actions: run, jump, duck), trained with **CMA-ES** (`train_controller.py`).

## Pipeline

```bash
python collect_data.py          # random rollouts from dino_env.py
python training_vae.py
python encode_data.py           # frames -> latents
python train_mdn_rnn.py
python collect_policy_rollouts.py  # / collect_blind_episodes.py: more data from the current policy
python train_controller.py      # CMA-ES
python evaluate_agent.py
```

`dino_env.py` is the game environment, and `test_live.py` runs the controller on the live game. `blog_assets/` holds the scripts that make the architecture diagram and training plots.

The second controller iteration reached a best CMA-ES fitness of about 284 (see `controller_train_iter2.log`).

## Files not in the repo

Collected data (`data/`, `latent_data/`) and per-epoch VAE checkpoints are git-ignored. The final `vae_final.pth` and `mdnrnn_final.pth` are included.
