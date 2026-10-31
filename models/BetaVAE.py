"""Beta-VAE model with a configurable KL-divergence weight.

A Beta-VAE uses the same encoder, decoder, Gaussian latent posterior, and
standard-normal prior as a VAE, but scales the KL term by ``beta``. Setting
``beta=1`` recovers the standard VAE objective; larger values emphasize prior
regularization and smaller values emphasize reconstruction.
"""

from math import isfinite
from numbers import Real

import torch

from models.VAE import VAE


class BetaVAE(VAE):
    """A VAE whose KL-divergence term is scaled by a non-negative ``beta``."""

    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, beta: float = 1.0
    ) -> None:
        """Initialize the inherited VAE architecture and validate ``beta``."""
        if isinstance(beta, bool) or not isinstance(beta, Real):
            raise ValueError("beta must be a finite, non-negative real number.")
        if not isfinite(float(beta)) or beta < 0:
            raise ValueError("beta must be a finite, non-negative real number.")
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.beta = float(beta)

    def loss_components(
        self,
        x: torch.Tensor,
        reconstructed_x: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-observation reconstruction, KL, and beta-weighted loss."""
        reconstruction_loss = self.criterion(x, reconstructed_x) / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar)) / x.size(0)
        total_loss = reconstruction_loss + self.beta * kl_loss
        return reconstruction_loss, kl_loss, total_loss

    def loss(
        self,
        x: torch.Tensor,
        reconstructed_x: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Return the beta-weighted VAE loss, normalized by batch size."""
        return self.loss_components(x, reconstructed_x, mean, logvar)[2]
