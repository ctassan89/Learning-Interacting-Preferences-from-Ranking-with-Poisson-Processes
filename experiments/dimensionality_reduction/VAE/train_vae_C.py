"""VAE_C: production blocks, v2 dimension 771; reconstruction plus KL; outputs results/dimensionality_reduction/VAE_C."""

from experiments.dimensionality_reduction.common import FEATURE_CONFIGURATIONS, run_vae

if __name__ == "__main__":
    run_vae("VAE_C", FEATURE_CONFIGURATIONS["C"])
