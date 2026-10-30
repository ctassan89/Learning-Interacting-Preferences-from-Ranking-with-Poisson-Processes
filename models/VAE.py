
import torch
import torch.nn as nn
from models.AE import decoder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class VAEencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Two outputs: mean and log-variance
        self.fc_mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.network(x)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = VAEencoder(input_dim, hidden_dim, latent_dim)
        self.decoder = decoder(latent_dim, hidden_dim, input_dim)
        self.criterion = nn.MSELoss(reduction="sum")


    def reparameterize(self, mean, logvar):
        # Convert log-variance to standard deviation
        std = torch.exp(0.5 * logvar)

        # Sample epsilon from N(0, I)
        eps = torch.randn_like(std)

        # Reparameterization trick
        z = mean + std * eps

        return z

    def forward(self, x):
        # Obtain distribution parameters
        mean, logvar = self.encoder(x)

        # Sample latent representation
        z = self.reparameterize(mean, logvar)

        # Reconstruct input
        reconstructed_x = self.decoder(z)

        return reconstructed_x, mean, logvar

    def loss(self, x, reconstructed_x, mean, logvar):
        reconstruction_loss = self.criterion(x, reconstructed_x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
        total_loss = (reconstruction_loss + kl_loss) / x.size(0)
        return total_loss



def train(model, train_dataloader, val_dataloader, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x in train_dataloader:
            optimizer.zero_grad()

            reconstructed_x, mean, logvar = model(x)
            loss = model.loss(x, reconstructed_x, mean, logvar)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        avg_loss = epoch_loss / len(train_dataloader.dataset)
        train_losses.append(avg_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x in val_dataloader:
                reconstructed_x, mean, logvar = model(x)
                loss = model.loss(x, reconstructed_x, mean, logvar)
                val_loss += loss.item() * x.size(0)

        val_avg_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(val_avg_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_loss:.6f}, "
            f"Val Loss: {val_avg_loss:.6f}"
        )

    return train_losses, val_losses




if __name__ == "__main__":

    # Synthetic dataset: 1000 observations, 500 features
    torch.manual_seed(42)

    Z = torch.randn(1000, 10)
    A = torch.randn(500, 10)
    eps = torch.randn(1000, 500) * 0.1

    X = Z @ A.T + eps  # Shape: (1000, 500)

    # Train/validation split: 80/20
    train_dataset, val_dataset = random_split(X, [800, 200])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    # Initialize VAE
    model = VAE(
        input_dim=500,
        hidden_dim=128,
        latent_dim=32
    )

    # Train VAE
    train_losses, val_losses = train(
        model,
        train_dataloader,
        val_dataloader,
        epochs=100,
        lr=0.001
    )

    # Verify output dimensions
    model.eval()
    with torch.no_grad():
        reconstructed_x, mean, logvar = model(X[:10])

    print("Input shape:", X[:10].shape)
    print("Reconstructed shape:", reconstructed_x.shape)
    print("Mean shape:", mean.shape)
    print("Logvar shape:", logvar.shape)

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("VAE Loss (Reconstruction + KL)")
    plt.title("VAE Training vs Validation")
    plt.legend()
    plt.grid(True)

    plt.savefig("vae_training_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
