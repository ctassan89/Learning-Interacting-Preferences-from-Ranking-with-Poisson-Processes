"""BetaVAE_D: all v2 blocks including collection and numeric (902 dimensions).

Uses the 32-dimensional Beta-VAE reconstruction plus beta-weighted KL objective
on the fixed split; outputs results/BetaVAE_D/beta_<value>.
"""

from experiments.common import FEATURE_CONFIGURATIONS, parse_beta_argument, run_beta_vae


if __name__ == "__main__":
    run_beta_vae("BetaVAE_D", FEATURE_CONFIGURATIONS["D"], parse_beta_argument())
