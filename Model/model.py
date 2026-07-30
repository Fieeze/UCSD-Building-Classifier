"""CNN model for classifying UCSD buildings."""

import re
import shutil
from pathlib import Path

# --- core tensor / neural network ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- pretrained backbones, image datasets, augmentation ---
import torchvision
from torchvision import datasets, models, transforms

# --- skorch: wraps a PyTorch module as a scikit-learn estimator ---
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, ProgressBar
from skorch.helper import predefined_split

# --- metrics and cross-validation ---
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

# --- arrays, images, plotting ---
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- config ---

NUM_CLASSES = 4          # how many buildings you are classifying
# 256 (not 224) so the final feature map is 16x16, which divides evenly by the
# pool size of 4. MPS cannot do adaptive pooling when the sizes do not divide.
IMAGE_SIZE = 256          # every photo is resized to IMAGE_SIZE x IMAGE_SIZE
BATCH_SIZE = 32
MAX_EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TRAIN_DIR = "Dataset/train"
TEST_DIR = "Dataset/test"
WEIGHTS = "Model/best_model.pt"


def _clean_sync_conflicts(directory):
    """Remove iCloud "<Building> 2" conflict-copy folders.

    Dataset/train and Dataset/test live under an iCloud-synced Desktop, so
    the sync daemon can drop in a numbered duplicate of a class folder at any
    time, independent of when split_dataset.py last ran. ImageFolder treats
    every subfolder as a class, so a stray empty duplicate crashes loading.
    Any folder named "<sibling name> <digits>" is such a duplicate, never
    real data, so it is safe to remove on every import.
    """
    path = Path(directory)
    if not path.is_dir():
        return
    names = {p.name for p in path.iterdir() if p.is_dir()}
    for name in names:
        match = re.match(r"^(.+) \d+$", name)
        if match and match.group(1) in names:
            shutil.rmtree(path / name)


for _dir in (TRAIN_DIR, TEST_DIR):
    _clean_sync_conflicts(_dir)


# --------------------------------------------------------- preprocessing ---
# Defined here so train.py and test.py import the SAME objects. If the two
# ever differ, accuracy becomes meaningless without anything visibly failing.

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),          # a mirrored building is still that building
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# No randomness — evaluation has to be repeatable.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ----------------------------------------------------------------- model ---

class SimpleCNN(nn.Module):
    """4 conv blocks, then a small MLP head.

    Input:  (batch, 3, 224, 224)
    Output: (batch, num_classes) logits — no softmax, CrossEntropyLoss adds it.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # block 1: 3 -> 32 channels,  224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2: 32 -> 64 channels,  112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3: 64 -> 128 channels,  56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4: 128 -> 256 channels,  28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Shrink each feature map to a 4x4 grid -> (batch, 256, 4, 4).
        # Keeps coarse spatial info (top/bottom/left/right) instead of
        # averaging it all away like AdaptiveAvgPool2d(1) would.
        self.pool = nn.AdaptiveAvgPool2d(4)

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (batch, 256*4*4) = (batch, 4096)
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ------------------------------------------------------- skorch estimator ---

net = NeuralNetClassifier(
    module=SimpleCNN,               #CNN(Previously defined)
    module__num_classes=NUM_CLASSES,# NUM_CLASSES = 4

    criterion=nn.CrossEntropyLoss,  
    optimizer=optim.AdamW,

    lr=LEARNING_RATE,               #LEARNING RATE = 0.001
    max_epochs=MAX_EPOCHS,          #MAX_EPOCHS = 40
    batch_size=BATCH_SIZE,          #BATCH_SIZE = 32

    iterator_train__shuffle=True,
    device=DEVICE,
)


if __name__ == "__main__":
    model = SimpleCNN()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    print("output shape:", tuple(model(dummy).shape))
    print("parameters  :", sum(p.numel() for p in model.parameters()))
    print("device      :", DEVICE)
