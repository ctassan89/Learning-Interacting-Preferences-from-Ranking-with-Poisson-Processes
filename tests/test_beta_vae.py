"""Synthetic tests for the Beta-VAE loss extension."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from experiments.common import (
    _beta_vae_batch_losses,
    beta_results_dir,
    ensure_empty_results_dir,
    normalize_beta,
    parse_beta_argument,
)
from models.BetaVAE import BetaVAE
from models.VAE import VAE


class BetaVAETests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.x = torch.randn(5, 7)
        self.mean = torch.randn(5, 3)
        self.logvar = torch.randn(5, 3) * 0.1
        self.reconstruction = torch.randn(5, 7)

    def test_inherits_vae_and_forward_shapes(self):
        model = BetaVAE(7, 8, 3)
        self.assertIsInstance(model, VAE)
        reconstructed, mean, logvar = model(self.x)
        self.assertEqual(reconstructed.shape, self.x.shape)
        self.assertEqual(mean.shape, (5, 3))
        self.assertEqual(logvar.shape, (5, 3))

    def test_beta_one_matches_standard_vae_loss(self):
        standard = VAE(7, 8, 3)
        beta_vae = BetaVAE(7, 8, 3, beta=1.0)
        self.assertTrue(
            torch.allclose(
                standard.loss(self.x, self.reconstruction, self.mean, self.logvar),
                beta_vae.loss(self.x, self.reconstruction, self.mean, self.logvar),
            )
        )

    def test_beta_zero_removes_kl_and_components_sum(self):
        model = BetaVAE(7, 8, 3, beta=0.0)
        reconstruction, kl, total = model.loss_components(
            self.x, self.reconstruction, self.mean, self.logvar
        )
        self.assertTrue(torch.allclose(total, reconstruction))
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_beta_multiplier_and_gradients(self):
        beta = 2.5
        model = BetaVAE(7, 8, 3, beta=beta)
        reconstruction, kl, total = model.loss_components(
            self.x, self.reconstruction, self.mean, self.logvar
        )
        self.assertTrue(torch.allclose(total, reconstruction + beta * kl))
        train_input = self.x.clone().requires_grad_()
        output, mean, logvar = model(train_input)
        model.loss(train_input, output, mean, logvar).backward()
        self.assertIsNotNone(train_input.grad)
        self.assertTrue(torch.isfinite(train_input.grad).all())
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )

    def test_supports_other_dimensions_and_rejects_invalid_beta(self):
        model = BetaVAE(11, 10, 4, beta=3)
        reconstructed, mean, logvar = model(torch.randn(2, 11))
        self.assertEqual(reconstructed.shape, (2, 11))
        self.assertEqual(mean.shape, (2, 4))
        self.assertEqual(logvar.shape, (2, 4))
        for beta in (-1, float("nan"), float("inf"), "one", True):
            with self.assertRaises(ValueError):
                BetaVAE(7, 8, 3, beta=beta)

    def test_beta_argument_and_stable_result_directory_names(self):
        self.assertEqual(parse_beta_argument([]), 1.0)
        self.assertEqual(parse_beta_argument(["--beta", "2"]), 2.0)
        self.assertEqual(normalize_beta(1), "1")
        self.assertEqual(normalize_beta(1.0), "1")
        self.assertEqual(normalize_beta(0.5), "0.5")
        self.assertEqual(
            beta_results_dir("BetaVAE_A", 1), beta_results_dir("BetaVAE_A", 1.0)
        )
        for beta in (-0.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                normalize_beta(beta)

    def test_result_directory_overwrite_protection_and_beta_batch_losses(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "beta_1"
            ensure_empty_results_dir(output)
            output.mkdir()
            ensure_empty_results_dir(output)
            (output / "model.pt").touch()
            with self.assertRaises(FileExistsError):
                ensure_empty_results_dir(output)

        model = BetaVAE(7, 8, 3, beta=2.0)
        reconstruction, kl, total = _beta_vae_batch_losses(model, self.x)
        self.assertTrue(torch.allclose(total, reconstruction + 2.0 * kl))


if __name__ == "__main__":
    unittest.main()
