"""
Variational Autoencoder 

Code inspired by MNIST Autoencoder using CNN from lecture code
Code implemented on Google Colab using T4 GPU
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

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from torchvision.utils import make_grid

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 128
LEARNING_RATE = 1e-4 # smaller, more stable
NUM_EPOCHS = 50  # increased training amount
BETA = 0.1  # weighting for KL divergence term

from google.colab import drive
drive.mount('/content/gdrive')

# ----------------------------
# Custom Dataset Class
# ----------------------------
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Transform to apply on an image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # convert to grayscale (1 channel)

        if self.transform:
            image = self.transform(image)

        return image   # no labels, so return only the image
    
    # Path to Google Drive dataset
train_dir = "/content/gdrive/MyDrive/keras_png_slices_data/keras_png_slices_train"
test_dir  = "/content/gdrive/MyDrive/keras_png_slices_data/keras_png_slices_test"

# Transform (convert to tensor + normalize)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create datasets
trainset = BrainMRIDataset(root_dir=train_dir, transform=transform)
testset = BrainMRIDataset(root_dir=test_dir, transform=transform)

# Create dataloaders
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")

class CNNVAE(nn.Module):
    def __init__(self, latent_dim=32, input_shape=(1, 256, 256)):
        super(CNNVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # (B, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (B, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # (B, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# (B, 256, 16, 16)
            nn.ReLU(),
        )

        # Dynamically compute flatten dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            h = self.encoder(dummy)
            self.h_shape = h.shape[1:]               # (256, 16, 16)
            self.flatten_dim = h.view(1, -1).size(1) # 256*16*16 = 65536

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # (B, 128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (B, 64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # (B, 32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # (B, 1, 256, 256)
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), *self.h_shape)  # (B, 256, 16, 16)
        out = self.decoder(h_dec)
        return out, mu, logvar

    def decode(self, z):
        """
        Decode a latent vector z to image
        """
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), *self.h_shape)  # use self.h_shape (256,16,16)
        out = self.decoder(h_dec)
        return out


def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    # Reconstruction loss (MSE better for grayscale MRIs)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD, recon_loss, KLD

import matplotlib.pyplot as plt
# Train the VAE
vae = CNNVAE(latent_dim=32, input_shape=(1, 64, 64)).to(device)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    vae_optimizer, mode='min', factor=0.5, patience=5
)

print("Training VAE...")
for epoch in range(NUM_EPOCHS):
    vae.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        recon_images, mu, logvar = vae(images)

        # Loss
        loss, recon_loss, kld = vae_loss_function(recon_images, images, mu, logvar, beta=BETA)

        # Backward
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()

    # Average losses
    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)

    # Scheduler update
    scheduler.step(avg_loss)

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] '
          f'Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}')

    # Visualize reconstructions
    vae.eval()
    with torch.no_grad():
        test_images = next(iter(test_loader)).to(device)
        recon_images, _, _ = vae(test_images)

        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            # Original
            axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_ylabel('Original', fontsize=12)

            # Reconstructed
            axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_ylabel('Reconstructed', fontsize=12)

        plt.suptitle(f'VAE Reconstructions - Epoch {epoch+1}', fontsize=14)
        plt.tight_layout()
        plt.show()

def plot_manifold(vae, n_grid=15, figsize=(15, 15)):
    """
    Plot a 2D manifold by sampling from a grid in latent space using PyTorch
    """
    vae.eval()

    # Create a grid in latent space
    grid_x = np.linspace(-3, 3, n_grid)
    grid_y = np.linspace(-3, 3, n_grid)

    fig, axes = plt.subplots(n_grid, n_grid, figsize=figsize)

    # Pick two random latent dimensions
    dim1 = np.random.randint(0, vae.latent_dim)
    dim2 = np.random.randint(0, vae.latent_dim)
    while dim2 == dim1:
        dim2 = np.random.randint(0, vae.latent_dim)

    print(f"Using latent dimensions {dim1} and {dim2} for manifold")

    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                # Latent vector: only vary dim1 & dim2, keep others = 0
                z = torch.zeros(1, vae.latent_dim, device=device)
                z[0, dim1] = torch.tensor(x, device=device)
                z[0, dim2] = torch.tensor(y, device=device)

                decoded_img = vae.decode(z)

                # Format image for plotting
                img = decoded_img.squeeze().cpu().numpy()
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')

    plt.suptitle(f'2D Manifold (Latent dims {dim1}, {dim2})', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_manifold(vae, n_grid=15)
