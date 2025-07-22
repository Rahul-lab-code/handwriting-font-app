# filepath: handwriting-font-app/handwriting-font-app/src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from dataset import HandwritingDataset
from model import CVAE
import pandas as pd
import os

def train_model(csv_file, root_dir, latent_dim=32, num_classes=62, batch_size=64, epochs=20, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(5, translate=(0.02, 0.02)),
        transforms.ToTensor(),
    ])
    
    dataset = HandwritingDataset(csv_file, root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_function(recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    print("[INFO] Starting training")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images, labels)
            loss = loss_function(recon, images, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}] [Batch {batch_idx}] Loss: {loss.item():.2f}")

        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.2f}")

    print("[INFO] Training complete")
    return model

if __name__ == "__main__":
    csv_file = 'data/labels.csv'
    root_dir = 'data/segmented_characters'
    model = train_model(csv_file, root_dir)
    torch.save(model.state_dict(), "handwriting_cvae.pth")
    print("[INFO] Model saved as handwriting_cvae.pth")