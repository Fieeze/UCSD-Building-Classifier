"""Grid search over learning rate and batch size, evaluated by cross-validation
on Dataset/train only (Dataset/test stays untouched for final evaluation).

    python Model/grid_search.py

Each candidate is trained for GRID_EPOCHS (fewer than train.py's MAX_EPOCHS)
since this fits len(param_grid combos) * cv models — the full epoch count is
for the final model in train.py, once you've picked a winner here.
"""

import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import GridSearchCV

from model import net, train_transform, NUM_CLASSES, TRAIN_DIR

GRID_EPOCHS = 5
CV_FOLDS = 3

PARAM_GRID = {
    "lr": [0.01, 0.001, 0.0001],
    "batch_size": [16, 32],
}

train_folder = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

if len(train_folder.classes) != NUM_CLASSES:
    raise SystemExit(
        f"NUM_CLASSES is {NUM_CLASSES} but {TRAIN_DIR} has "
        f"{len(train_folder.classes)} building(s): {train_folder.classes}\n"
        f"Set NUM_CLASSES = {len(train_folder.classes)} in Model/model.py."
    )

print(f"loading {len(train_folder)} training images into memory...")
X_train = torch.stack([image for image, _ in train_folder])
y_train = np.array(train_folder.targets)

# Quiet per-epoch printing so the search output stays readable across every
# candidate x fold combination; GridSearchCV reports its own summary after.
net.set_params(max_epochs=GRID_EPOCHS, verbose=0)

search = GridSearchCV(
    net,
    PARAM_GRID,
    cv=CV_FOLDS,
    scoring="accuracy",
    refit=False,   # train.py trains the real final model at full MAX_EPOCHS
    verbose=2,
)

print(f"searching {len(PARAM_GRID['lr']) * len(PARAM_GRID['batch_size'])} "
      f"combinations x {CV_FOLDS} folds, {GRID_EPOCHS} epochs each...")
search.fit(X_train, y_train)

print("\nbest params:", search.best_params_)
print("best cv accuracy:", search.best_score_)

print(f"\n{'lr':>10}{'batch_size':>12}{'mean accuracy':>16}{'std':>8}")
results = search.cv_results_
for lr, bs, mean, std in zip(results["param_lr"], results["param_batch_size"],
                             results["mean_test_score"], results["std_test_score"]):
    print(f"{lr:>10}{bs:>12}{mean:>16.4f}{std:>8.4f}")

print(f"\nSet lr / batch_size in Model/model.py's net(...) to {search.best_params_}, "
      f"then run: python Model/train.py")
