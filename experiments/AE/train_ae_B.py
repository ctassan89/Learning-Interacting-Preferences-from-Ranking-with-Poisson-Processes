"""AE_B: creative blocks, v2 dimension 632; MSE Autoencoder; outputs results/AE_B."""

from experiments.common import FEATURE_CONFIGURATIONS, run_ae

if __name__ == "__main__":
    run_ae("AE_B", FEATURE_CONFIGURATIONS["B"])
