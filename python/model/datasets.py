import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class BubbleChamberDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Auto-discover all image files (assumes same name in both dirs)
        self.image_paths = sorted(self.image_dir.glob("*.png"))

        # Verify masks exist for every image
        for p in self.image_paths:
            mask_p = self.mask_dir / p.name
            if not mask_p.exists():
                raise FileNotFoundError(f"Missing mask for {p.name}")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_dir / self.image_paths[idx].name).convert("L")

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),       # pick your working resolution
    transforms.ToTensor(),               # [0, 1] float, shape (H, W)
])

# If you need to apply the same spatial transform to both img & mask:
class ImgMaskTransform:
    def __init__(self, size=512):
        self.size = size
    def __call__(self, img, mask):
        img = transforms.functional.resize(img, (self.size, self.size))
        mask = transforms.functional.resize(mask, (self.size, self.size),
                                             interpolation=transforms.InterpolationMode.NEAREST)
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        return img, mask

from torch.utils.data import random_split

dataset = BubbleChamberDataset("data/images", "data/masks", # MUST BE CHANGED!
                               transform=ImgMaskTransform(512))

# 90/10 split
n_train = int(0.9 * len(dataset))
train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

import matplotlib.pyplot as plt

img, mask = next(iter(train_dl))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img[0], cmap="gray"); plt.title("Image")
plt.subplot(1, 2, 2); plt.imshow(mask[0], cmap="gray"); plt.title("Mask")
plt.show()   