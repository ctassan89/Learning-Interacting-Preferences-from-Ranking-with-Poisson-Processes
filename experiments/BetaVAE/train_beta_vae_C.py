"""BetaVAE_C: B plus companies and language (v2: 771 dimensions).

Uses the 32-dimensional Beta-VAE reconstruction plus beta-weighted KL objective
on the fixed split; outputs results/BetaVAE_C/beta_<value>.
"""

from experiments.common import FEATURE_CONFIGURATIONS, parse_beta_argument, run_beta_vae


if __name__ == "__main__":
    run_beta_vae("BetaVAE_C", FEATURE_CONFIGURATIONS["C"], parse_beta_argument())
