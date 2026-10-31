"""AE_D: complete v2 blocks, dimension 902; MSE Autoencoder; outputs results/AE_D."""

from experiments.common import FEATURE_CONFIGURATIONS, run_ae

if __name__ == "__main__":
    run_ae("AE_D", FEATURE_CONFIGURATIONS["D"])
