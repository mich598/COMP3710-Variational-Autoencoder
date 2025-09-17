"""
Variational Autoencoder (VAE) on OASIS preprocessed dataset
Runs cleanly on GPU cluster (Rangpur, A100).

Code based on lecture example for building a CNN-based VAE using MNIST dataset
"""

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BETA = 1.0  # weighting for KL divergence term

# ----------------------------
# Dataset
# ----------------------------
print("> Setting up dataset")

transform = transforms.Compose([
    transforms.ToTensor(),   # converts [0,255] -> [0,1]
])

# Load dataset
trainset = ImageFolder(root='/home/groups/comp3710/OASIS/keras_png_slices_train', transform=transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

testset = ImageFolder(root='/home/groups/comp3710/OASIS/keras_png_slices_test', transform=transform)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

class CNNVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (same as before but outputs mean and log_var)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 7x7 -> 4x4

            nn.Flatten(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # Log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Reshape to (batch, 128, 4, 4)

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),  # 16x16 -> 28x28
            nn.Sigmoid()  # Output in [0, 1] for image reconstruction
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0,1)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean for inference. Using mu (mean) rather than sampling from the full distribution because:
                        # It provides deterministic, reproducible results
                        # The mean represents the "most likely" latent representation
                        # It avoids noise that could make interpolation less smooth

    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    beta: weight for KL divergence (beta-VAE)
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD, BCE, KLD

# ----------------------------
# VAE Loss
# ----------------------------
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD

# ----------------------------
# Train VAE
# ----------------------------
vae = CNNVAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

print("Training VAE...")
for epoch in range(NUM_EPOCHS):
    vae.train()
    total_loss, total_bce, total_kld = 0, 0, 0

    for images, _ in train_loader:
        images = images.to(device)

        recon_images, mu, logvar = vae(images)
        loss, bce, kld = vae_loss_function(recon_images, images, mu, logvar, beta=BETA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_bce = total_bce / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} BCE: {avg_bce:.4f} KLD: {avg_kld:.4f}')

# Create a 2D manifold visualization of scans
print("Creating 2D manifold of scans...")

def plot_digit_manifold(vae, n_grid=15, figsize=(15, 15)):
    """
    Plot a 2D manifold of digits by sampling from a grid in latent space
    """
    vae.eval()

    # Create a grid of points in 2D latent space
    # We'll use the first 2 dimensions of the latent space
    grid_x = np.linspace(-3, 3, n_grid)
    grid_y = np.linspace(-3, 3, n_grid)

    fig, axes = plt.subplots(n_grid, n_grid, figsize=figsize)

    # Randomly select which 2 dimensions to use for the manifold
    dim1 = np.random.randint(0, vae.latent_dim)
    dim2 = np.random.randint(0, vae.latent_dim)
    while dim2 == dim1:
        dim2 = np.random.randint(0, vae.latent_dim)

    print(f"Using latent dimensions {dim1} and {dim2} for manifold")

    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                # Create latent vector with small random noise in other dimensions
                z = torch.randn(1, vae.latent_dim).to(device) * 0.1  # Small noise
                z[0, dim1] = x
                z[0, dim2] = y

                # Decode the latent vector
                decoded_img = vae.decode(z)

                # Plot the decoded image
                axes[i, j].imshow(decoded_img.cpu().squeeze(), cmap='gray')
                axes[i, j].axis('off')

    plt.suptitle('2D Manifold of Digits (Latent Space Dimensions 0 & 1)', fontsize=16)
    plt.tight_layout()
    plt.show()

# Run manifold visualization
manifold = plot_digit_manifold(vae, n_grid=15)
plt.imsave("vae_manifold.png", manifold)
