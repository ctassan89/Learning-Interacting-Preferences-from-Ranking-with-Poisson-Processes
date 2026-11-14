"""AE_B: creative blocks, v2 dimension 632; MSE Autoencoder; outputs results/dimensionality_reduction/AE_B."""

from experiments.dimensionality_reduction.common import FEATURE_CONFIGURATIONS, run_ae

if __name__ == "__main__":
    run_ae("AE_B", FEATURE_CONFIGURATIONS["B"])
