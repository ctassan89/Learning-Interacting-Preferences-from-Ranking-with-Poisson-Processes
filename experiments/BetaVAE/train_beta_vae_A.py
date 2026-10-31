"""BetaVAE_A: genres, keywords, text (v2: 248 dimensions).

Uses the 128-64-latent(32)-64-128 Beta-VAE and reconstruction plus beta-weighted
KL loss on the fixed 6,945/1,226/854 split; outputs results/BetaVAE_A/beta_<value>.
"""

from experiments.common import FEATURE_CONFIGURATIONS, parse_beta_argument, run_beta_vae


if __name__ == "__main__":
    run_beta_vae("BetaVAE_A", FEATURE_CONFIGURATIONS["A"], parse_beta_argument())
