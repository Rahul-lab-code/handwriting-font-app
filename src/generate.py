import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import CVAE
from dataset import ALL_CLASSES  # Use the same class order as training

def generate_characters(model, num_samples=10, output_dir='./data/generated_characters'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    print("[INFO] Generating new characters")
    
    with torch.no_grad():
        for idx, char in enumerate(ALL_CLASSES):
            for i in range(num_samples):
                y = torch.tensor([idx]).to(next(model.parameters()).device)
                z = torch.randn(1, model.latent_dim).to(next(model.parameters()).device)
                gen_image = model.decode(z, y).cpu().squeeze().numpy()
                gen_image = (gen_image * 255).astype(np.uint8)
                plt.imsave(os.path.join(output_dir, f'generated_{char}_{i}.png'), gen_image, cmap='gray')
    print("[INFO] Character generation complete")



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(ALL_CLASSES)  # Ensure this matches training!
    model = CVAE(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('handwriting_cvae.pth', map_location=device))
    generate_characters(model, num_samples=10)