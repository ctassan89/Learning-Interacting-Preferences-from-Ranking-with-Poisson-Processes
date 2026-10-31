"""VAE_B: creative blocks, v2 dimension 632; reconstruction plus KL; outputs results/VAE_B."""

from experiments.common import FEATURE_CONFIGURATIONS, run_vae

if __name__ == "__main__":
    run_vae("VAE_B", FEATURE_CONFIGURATIONS["B"])
