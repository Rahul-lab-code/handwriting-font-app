from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

# Define all possible classes in the order you want
ALL_CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
label_to_idx = {c: i for i, c in enumerate(ALL_CLASSES)}

class HandwritingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_path).convert('L')
        label_char = str(self.labels_df.iloc[idx, 1])
        # Map the character to its index, fallback to 0 if not found (or raise error)
        label = label_to_idx.get(label_char, 0)
        if self.transform:
            image = self.transform(image)
        return image, label