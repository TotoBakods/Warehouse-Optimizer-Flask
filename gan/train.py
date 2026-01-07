import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from data_loader import get_dataloader
from model import Generator, Discriminator
import pickle

# Parameters
LATENT_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.0002
B1 = 0.5
B2 = 0.999

# Get absolute path to datasets.csv assuming it is in the parent directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, '..', 'datasets.csv')
CHECKPOINT_DIR = os.path.join(current_dir, 'checkpoints')
SCALER_PATH = os.path.join(current_dir, 'scaler.pkl')

def train():
    # Setup directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print(f"Loading data from {DATA_FILE}...")
    dataloader, scaler = get_dataloader(DATA_FILE, BATCH_SIZE)
    
    # Save scaler for generation
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Model
    generator = Generator(LATENT_DIM, 4).to(device)
    discriminator = Discriminator(4).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(B1, B2))
    
    # Loss
    adversarial_loss = nn.BCELoss()
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        for i, imgs in enumerate(dataloader):
            
            # Configure input
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Log progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        
        # Save checkpoint periodically
        if epoch % 5 == 0:
             torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
             torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator.pth"))

    # Final save
    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
    print("Training finished. Model saved.")

if __name__ == "__main__":
    train()
