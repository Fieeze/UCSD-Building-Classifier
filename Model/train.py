"""Train the model on Dataset/train and validate on Dataset/test, then save
the weights.

    python Model/train.py

X_train/y_train hold all four buildings' training photos as one stacked
tensor and one label array; X_valid/y_valid hold all four buildings' held-out
test photos the same way (ImageFolder treats each building's subfolder as a
class). Passing X_valid/y_valid through predefined_split makes skorch report
accuracy on those real held-out photos every epoch, instead of splitting off
part of the training set. Watch that number, not the training accuracy — if
training keeps rising while validation stalls, the model is memorizing
photos.

Stacking into tensors up front (rather than handing skorch the ImageFolder
directly) means train_transform's random flip/jitter is drawn once per photo
and reused every epoch, not re-randomized each time — a small trade against
having plain X/y arrays to inspect and feed the model directly.
"""

import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import TensorDataset
from skorch.helper import predefined_split

from model import net, train_transform, eval_transform, NUM_CLASSES, TRAIN_DIR, TEST_DIR, WEIGHTS

train_folder = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_folder = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

# A mismatch here trains a model whose outputs do not line up with the labels,
# so fail loudly instead of producing a quietly broken checkpoint.
if train_folder.classes != valid_folder.classes:
    raise SystemExit(
        f"Train/test class mismatch: train has {train_folder.classes}, "
        f"test has {valid_folder.classes}.\n"
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

print(f"loading {len(train_folder)} training images into memory...")
X_train = torch.stack([image for image, _ in train_folder])
y_train = np.array(train_folder.targets)

print(f"loading {len(valid_folder)} validation images into memory...")
X_valid = torch.stack([image for image, _ in valid_folder])
y_valid = np.array(valid_folder.targets)

print(f"training on {len(X_train)} images, validating on {len(X_valid)}, "
      f"{NUM_CLASSES} buildings: {train_folder.classes}")

valid_dataset = TensorDataset(X_valid, torch.as_tensor(y_valid, dtype=torch.long))
net.set_params(train_split=predefined_split(valid_dataset))
net.fit(X_train, y_train)
net.save_params(f_params=WEIGHTS)

print(f"\nweights saved to {WEIGHTS}")
print("next: python Model/test.py")
