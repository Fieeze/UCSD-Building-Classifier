"""Test the model on Dataset/test and print its accuracy.

    python Model/test.py
"""

from torchvision import datasets
from sklearn.metrics import accuracy_score

from model import net, eval_transform, TEST_DIR, WEIGHTS

dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)
y_true = [label for _, label in dataset.samples]

net.initialize()
net.load_params(f_params=WEIGHTS)
y_pred = net.predict(dataset)

print("images  :", len(dataset))
print("accuracy:", accuracy_score(y_true, y_pred))
