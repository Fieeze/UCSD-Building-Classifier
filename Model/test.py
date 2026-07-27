"""Test the model on a dataset and print its accuracy.

    python Model/test.py

Expects one folder per building:  Dataset/test/Geisel_Library/img001.jpg ...
"""

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

from model import net, IMAGE_SIZE

DATA_DIR = "Dataset/test"
WEIGHTS = "Model/best_model.pt"

# Training must use this exact transform, or the accuracy means nothing.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
y_true = [label for _, label in dataset.samples]

net.initialize()
net.load_params(f_params=WEIGHTS)
y_pred = net.predict(dataset)

print("images  :", len(dataset))
print("accuracy:", accuracy_score(y_true, y_pred))
