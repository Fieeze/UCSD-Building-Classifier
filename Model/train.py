"""Train the model on Dataset/train and save the weights.

    python Model/train.py

skorch holds out 20% of the training images to report validation accuracy each
epoch. Watch that number, not the training accuracy — if training keeps rising
while validation stalls, the model is memorizing photos.
"""

from torchvision import datasets

from model import net, train_transform, NUM_CLASSES, TRAIN_DIR, WEIGHTS

dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

# A mismatch here trains a model whose outputs do not line up with the labels,
# so fail loudly instead of producing a quietly broken checkpoint.
if len(dataset.classes) != NUM_CLASSES:
    raise SystemExit(
        f"NUM_CLASSES is {NUM_CLASSES} but {TRAIN_DIR} has "
        f"{len(dataset.classes)} building(s): {dataset.classes}\n"
        f"Set NUM_CLASSES = {len(dataset.classes)} in Model/model.py."
    )

if NUM_CLASSES < 2:
    raise SystemExit(
        "Only one building in the dataset — there is nothing to tell apart.\n"
        "The model would score 100% by always answering the same thing.\n"
        "Add photos for a second building to Dataset/raw/, then rerun split_dataset.py."
    )

print(f"training on {len(dataset)} images, {NUM_CLASSES} buildings: {dataset.classes}")

net.fit(dataset, y=None)
net.save_params(f_params=WEIGHTS)

print(f"\nweights saved to {WEIGHTS}")
print("next: python Model/test.py")
