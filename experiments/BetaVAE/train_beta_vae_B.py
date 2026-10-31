"""BetaVAE_B: A plus cast, directors, writers (v2: 632 dimensions).

Uses the 32-dimensional Beta-VAE reconstruction plus beta-weighted KL objective
on the fixed split; outputs results/BetaVAE_B/beta_<value>.
"""

from experiments.common import FEATURE_CONFIGURATIONS, parse_beta_argument, run_beta_vae


if __name__ == "__main__":
    run_beta_vae("BetaVAE_B", FEATURE_CONFIGURATIONS["B"], parse_beta_argument())
