import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
import os

class VOCDataset(Dataset):
    def __init__(self, root, year, image_set, transform=None):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)

        # Extract bounding boxes and labels
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['name'])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)  # For simplicity, labels are kept as strings, you might want to encode them as integers.

        return img, boxes, labels

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize dataset and dataloader
