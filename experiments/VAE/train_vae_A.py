"""VAE_A: content blocks, v2 dimension 248; reconstruction plus KL; outputs results/VAE_A."""

from experiments.common import FEATURE_CONFIGURATIONS, run_vae

if __name__ == "__main__":
    run_vae("VAE_A", FEATURE_CONFIGURATIONS["A"])
