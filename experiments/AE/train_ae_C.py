"""AE_C: production blocks, v2 dimension 771; MSE Autoencoder; outputs results/AE_C."""

from experiments.common import FEATURE_CONFIGURATIONS, run_ae

if __name__ == "__main__":
    run_ae("AE_C", FEATURE_CONFIGURATIONS["C"])
