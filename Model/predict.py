"""Predict the building in every photo dropped into Dataset/predict/.

    python Model/predict.py

Unlike train.py/test.py, this folder has no per-building subfolders — just
loose photos of unknown buildings — so it can't be read with ImageFolder.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets

from model import net, eval_transform, TRAIN_DIR, WEIGHTS

PREDICT_DIR = Path("Dataset/predict")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageFolder assigns class indices alphabetically by subfolder name — reading
# TRAIN_DIR here (rather than hardcoding the list) keeps this in sync with
# whatever training actually used.
classes = datasets.ImageFolder(TRAIN_DIR).classes

files = sorted(f for f in PREDICT_DIR.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
if not files:
    raise SystemExit(f"No images in {PREDICT_DIR}/. Drop some photos in there first.")

images = torch.stack([eval_transform(Image.open(f).convert("RGB")) for f in files])

net.initialize()
net.load_params(f_params=WEIGHTS)

probs = net.predict_proba(images)
predictions = probs.argmax(axis=1)

for f, pred, prob in zip(files, predictions, probs):
    print(f"{f.name:<30} {classes[pred]:<25} {prob[pred] * 100:.1f}%")
