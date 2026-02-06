import os
from PIL import Image
import torch
from torch.utils.data import Dataset


def _get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _normalize_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))

class SkinDs(Dataset):
    def __init__(self, df, label_map=None, transform=None, image_root=None):
        self.df = df
        self.transform = transform
        self.label_map = label_map or {diag: i for i, diag in enumerate(df['diagnosis'].unique())}
        base_dir = _get_base_dir()
        if image_root is None:
            image_root = os.path.join(base_dir, 'data', 'images')
        self.image_root = _normalize_path(image_root, base_dir)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_root, f"{row['image']}.jpg")).convert('RGB')
        label = self.label_map[row['diagnosis']]
        weight = row.get('weight', 1.0)
        skin_tone = row.get('skin_tone', 'Unknown')

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label), torch.tensor(weight, dtype=torch.float32), skin_tone
