import numpy as np
import cv2

# Load the first episode
data = np.load("data/episode_0000.npz")
frames = data["obs"]      # Shape: (T, 1, 64, 64)
actions = data["actions"] # Shape: (T, 3)

print(f"Episode length: {len(frames)} frames")
print(f"Observation array shape: {frames.shape}")

# Save the first 5 frames as PNGs to visually inspect them
for i in range(min(5, len(frames))):
    frame_2d = frames[i, 0] # Extract (64, 64)
    cv2.imwrite(f"sample_frame_{i}.png", frame_2d)

print("Saved sample frames (sample_frame_0.png to sample_frame_4.png). Check them!")