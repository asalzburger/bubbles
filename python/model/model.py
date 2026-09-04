import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BubbleChamberDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        mask = Image.open(self.masks[idx]).convert('L')
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

# Transform: resize, normalize, to tensor
class ToTensor:
    def __call__(self, img, mask):
        img = T.ToTensor()(img)       # [0,1] float, shape (H,W)
        mask = T.ToTensor()(mask)     # [0,1] float
        return img, mask

#
#   U-NET
#

import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

#
#   Training
#

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))  # per-pixel probability

from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()  # use without sigmoid if you prefer
# If using BCEWithLogitsLoss, remove sigmoid from model output

train_ds = BubbleChamberDataset(train_images, train_masks, transform=ToTensor())
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

for epoch in range(50):
    model.train()
    for images, masks in train_dl:
        images = images.unsqueeze(1).to(device)  # (B, 1, H, W)
        masks = masks.unsqueeze(1).to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#
#   Track Extraction
#

from skimage.morphology import skeletonize
import cv2

def extract_tracks(image_path, model, device):
    img = Image.open(image_path).convert('L')
    img_tensor = T.ToTensor()(img).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor).squeeze().cpu().numpy()

    # Threshold to binary mask
    mask = (pred > 0.5).astype(np.uint8) * 255

    # Skeletonize to get 1-pixel centerlines
    skeleton = skeletonize(mask > 0).astype(np.uint8) * 255

    # Find connected components (each = one track segment)
    num_labels, labels = cv2.connectedComponents(skeleton)

    tracks = []
    for label in range(1, num_labels):  # skip background (0)
        pts = np.column_stack(np.where(labels == label))  # (y, x) coords
        if len(pts) > 10:  # filter noise
            # Fit a circle (charged track in B-field)
            # Use algebraic circle fit
            x, y = pts[:, 1], pts[:, 0]
            A = np.column_stack([x, y, np.ones(len(x))])
            b = x**2 + y**2
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = result[0]/2, result[1]/2
            r = np.sqrt(result[2] + cx**2 + cy**2)
            tracks.append({'center': (cx, cy), 'radius': r, 'points': pts})

    return tracks, skeleton   