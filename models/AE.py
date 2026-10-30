
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split


class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.network(x)


class decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.network(z)


class autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = decoder(latent_dim, hidden_dim, input_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

    def loss(self, x, reconstructed_x):
        return self.criterion(reconstructed_x, x)


def train(model, train_dataloader, val_dataloader, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x in train_dataloader:
            optimizer.zero_grad()

            reconstructed_x = model(x)
            loss = model.loss(x, reconstructed_x)

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
                reconstructed_x = model(x)
                loss = model.loss(x, reconstructed_x)
                val_loss += loss.item() * x.size(0)
        val_avg_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(val_avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_avg_loss:.6f}")

    return train_losses, val_losses


if __name__ == "__main__":

    # Synthetic dataset: 1000 movies, 500 features each
    Z = torch.randn(1000, 10)
    A = torch.randn(500, 10)
    eps = torch.randn(1000, 500) * 0.1
    X = Z @ A.T + eps  # Shape: (1000, 500)
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

    # Initialize autoencoder
    model = autoencoder(
        input_dim=500,
        hidden_dim=128,
        latent_dim=32
    )

    # Train autoencoder
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
        x_hat = model(X[:10])
        z = model.encoder(X[:10])

    print("Input shape:", X[:10].shape)
    print("Embedding shape:", z.shape)
    print("Reconstructed shape:", x_hat.shape)


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
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training vs Validation")
    plt.legend()
    plt.grid(True)

    plt.savefig("ae_training_validation.png", dpi=150, bbox_inches="tight")
    plt.close()