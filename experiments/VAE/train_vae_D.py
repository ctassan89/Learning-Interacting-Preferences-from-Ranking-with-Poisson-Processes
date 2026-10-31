"""VAE_D: complete v2 blocks, dimension 902; reconstruction plus KL; outputs results/VAE_D."""

from experiments.common import FEATURE_CONFIGURATIONS, run_vae

if __name__ == "__main__":
    run_vae("VAE_D", FEATURE_CONFIGURATIONS["D"])
