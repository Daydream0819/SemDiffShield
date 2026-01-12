import os
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    Read .pt latent tensors from a folder.
    Each file should be a torch Tensor with shape [4, H, W] (e.g., [4,64,64]).
    """
    def __init__(self, root: str):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Latent folder not found: {root}")
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".pt")
        ])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files under: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = torch.load(self.files[idx])  # Tensor [4,H,W]
        if not torch.is_tensor(z):
            raise TypeError(f"Loaded object is not a torch.Tensor: {self.files[idx]}")
        if z.ndim != 3:
            raise ValueError(f"Latent must be 3D [C,H,W], got {tuple(z.shape)} from {self.files[idx]}")
        return z.float()
