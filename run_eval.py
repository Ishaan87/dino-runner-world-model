import numpy as np
from evaluate_agent import evaluate

scores = evaluate(num_episodes=50, verbose=False)
np.save("eval_scores.npy", scores)
print("Saved eval_scores.npy")
