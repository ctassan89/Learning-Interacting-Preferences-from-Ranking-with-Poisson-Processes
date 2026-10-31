"""AE_A: content blocks (genres, keywords, text), v2 dimension 248.

Uses the 128-64-32-64-128 Autoencoder with MSE on the fixed leakage-free
6,945/1,226/854 train/validation/held-out split. Outputs results/AE_A.
"""

from experiments.common import FEATURE_CONFIGURATIONS, run_ae

if __name__ == "__main__":
    run_ae("AE_A", FEATURE_CONFIGURATIONS["A"])
