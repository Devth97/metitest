
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
BATCH_SIZE = 128
LATENT_DIM = 100 # Size of the noise vector
NUM_EPOCHS = 50 # You can adjust this based on your GPU time and desired quality
LEARNING_RATE = 0.0002
BETA1 = 0.5 # Adam optimizer parameter
IMAGE_SIZE = 28 # MNIST image size
NUM_CHANNELS = 1 # Grayscale images

# Directory to save generated images and models
OUTPUT_DIR = 'gan_outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'generator_mnist_dcgan.pth')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize images to [-1, 1] as per GAN best practices
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model Architecture ---

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super(Generator, self).__init__()
        
        self.init_size = image_size // 4  # 7 for 28x28
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # Up to 14x14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Up to 28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z.view(z.size(0), -1))  # Flatten z, then linear
        out = out.view(out.size(0), 128, self.init_size, self.init_size)  # Reshape
        img = self.conv_blocks(out)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_channels, image_size):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: Image (num_channels, 28, 28)
            nn.Conv2d(num_channels, 64, 3, 2, 1, bias=False),  # (N, 64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # (N, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # (N, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),  # Global average pooling to (N, 256, 1, 1)
            nn.Conv2d(256, 1, 1, 1, 0, bias=False),  # (N, 1, 1, 1)
            nn.Sigmoid()  # Output probability
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)  # Flatten to (N)

# --- Initialization ---
netG = Generator(LATENT_DIM, NUM_CHANNELS, IMAGE_SIZE).to(device)
netD = Discriminator(NUM_CHANNELS, IMAGE_SIZE).to(device)

# Initialize weights to random normal
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# --- Loss Function and Optimizers ---
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# --- Training Loop ---
print("\nStarting Training Loop...")
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader, 0):
        # (1) Update Discriminator
        
        # Train with real images
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1., dtype=torch.float, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake images
        noise = torch.randn(b_size, LATENT_DIM, device=device)  # Fixed: 1D noise vector
        fake = netG(noise)
        label.fill_(0.)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update Generator
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] Epoch [{epoch}/{NUM_EPOCHS}] Batch [{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

# --- Save the Generator Model ---
torch.save(netG.state_dict(), MODEL_SAVE_PATH)
print(f"\nGenerator model saved to {MODEL_SAVE_PATH}")

# --- Test Generation ---
print("\nGenerating sample images for verification...")
num_samples = 25
sample_noise = torch.randn(num_samples, LATENT_DIM, device=device)  # Fixed: 1D noise
with torch.no_grad():
    generated_images = netG(sample_noise).cpu().numpy()

# Denormalize images from [-1, 1] to [0, 1]
generated_images = (generated_images + 1) / 2.0

plt.figure(figsize=(10,10))
for i in range(num_samples):
    plt.subplot(5, 5, i+1)
    plt.imshow(generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle(f"Generated Digits after {NUM_EPOCHS} Epochs", fontsize=16)
plt.tight_layout()
plt.show()

print("\nTraining completed successfully!")
