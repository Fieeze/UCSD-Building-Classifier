"""Train the model on 75% of Dataset/train, validating each epoch on the
other 25%, then save the weights. Dataset/test is never touched here — it is
reserved as a true held-out set for Model/test.py.

    python Model/train.py

The 25% validation slice is a stratified split of Dataset/train (same class
proportions as the full set), held out with eval_transform (no random
flip/jitter) so validation accuracy is measured on clean images, not
augmented ones. Passing it through predefined_split makes skorch report
accuracy on it every epoch instead of splitting off part of the training set
itself. Watch that number, not the training accuracy — if training keeps
rising while validation stalls, the model is memorizing photos.

Stacking into tensors up front (rather than handing skorch the ImageFolder
directly) means train_transform's random flip/jitter is drawn once per photo
and reused every epoch, not re-randomized each time — a small trade against
having plain X/y arrays to inspect and feed the model directly.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data import TensorDataset
from skorch.helper import predefined_split

from model import net, train_transform, eval_transform, NUM_CLASSES, TRAIN_DIR, TEST_DIR, WEIGHTS

VALID_FRACTION = 0.25
SEED = 42

# No transform yet — just scans Dataset/train for file paths and labels.
train_folder = datasets.ImageFolder(TRAIN_DIR)
test_folder = datasets.ImageFolder(TEST_DIR)

# A mismatch here trains a model whose outputs do not line up with the labels,
# so fail loudly instead of producing a quietly broken checkpoint.
if train_folder.classes != test_folder.classes:
    raise SystemExit(
        f"Train/test class mismatch: train has {train_folder.classes}, "
        f"test has {test_folder.classes}.\n"
        f"Rerun split_dataset.py to rebuild both from Dataset/raw/."
    )

if len(train_folder.classes) != NUM_CLASSES:
    raise SystemExit(
        f"NUM_CLASSES is {NUM_CLASSES} but {TRAIN_DIR} has "
        f"{len(train_folder.classes)} building(s): {train_folder.classes}\n"
        f"Set NUM_CLASSES = {len(train_folder.classes)} in Model/model.py."
    )

if NUM_CLASSES < 2:
    raise SystemExit(
        "Only one building in the dataset — there is nothing to tell apart.\n"
        "The model would score 100% by always answering the same thing.\n"
        "Add photos for a second building to Dataset/raw/, then rerun split_dataset.py."
    )

train_idx, valid_idx = train_test_split(
    range(len(train_folder.samples)),
    test_size=VALID_FRACTION,
    stratify=train_folder.targets,
    random_state=SEED,
)


def load(indices, transform):
    images, labels = [], []
    for i in indices:
        path, label = train_folder.samples[i]
        images.append(transform(train_folder.loader(path)))
        labels.append(label)
    return torch.stack(images), np.array(labels)


print(f"loading {len(train_idx)} training images into memory...")
X_train, y_train = load(train_idx, train_transform)

print(f"loading {len(valid_idx)} validation images into memory "
      f"({VALID_FRACTION:.0%} of Dataset/train, held out)...")
X_valid, y_valid = load(valid_idx, eval_transform)

print(f"training on {len(X_train)} images, validating on {len(X_valid)}, "
      f"{NUM_CLASSES} buildings: {train_folder.classes}")
print(f"Dataset/test ({len(test_folder)} images) untouched — reserved for Model/test.py")

valid_dataset = TensorDataset(X_valid, torch.as_tensor(y_valid, dtype=torch.long))
net.set_params(train_split=predefined_split(valid_dataset))
net.fit(X_train, y_train)
net.save_params(f_params=WEIGHTS)

print(f"\nweights saved to {WEIGHTS}")
print("next: python Model/test.py")
