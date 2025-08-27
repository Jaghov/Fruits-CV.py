import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from train import train
from test import test
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = "fruits/train"

# Define transforms (resize + tensor + normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load full full_dataset from train/ (since only that has labels)
full_dataset = datasets.ImageFolder(root=ds, transform=transform)

# Check number of classes
classes = full_dataset.classes
num_classes = len(classes)
targets = full_dataset.targets  # class indices


# Check the number of items per class
for class_name in os.listdir(ds):
    class_path = os.path.join(ds, class_name)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        print(f"{class_name}, {num_images}")

# 70% train, 15% val, 15% test
train_idx, temp_idx = train_test_split(
    range(len(targets)),
    test_size=0.3,
    stratify=targets,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=[targets[i] for i in temp_idx],
    random_state=42
)

# Wrap back into Subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)
test_dataset  = Subset(full_dataset, test_idx)

# Wrap in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Number of training samples:", len(train_dataset))
print("Number of testing samples:", len(test_dataset))
print("Number of classes:", num_classes)



model, train_losses, val_losses, train_accuracies, val_accuracies = train(num_classes=num_classes, train_loader=train_loader, val_loader=val_loader, dataset_size=len(train_idx), device=device)

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12,5))

# ---- Loss plot ----
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label="Train Loss")
plt.plot(epochs, val_losses, 'r-', label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

# ---- Accuracy plot ----
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b-', label="Train Accuracy")
plt.plot(epochs, val_accuracies, 'r-', label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()

# ---- Save figure to disk ----
plt.savefig("training_curve.png", dpi=300)  # high-res PNG
plt.show()  # optional: also display

test(model, test_loader, device=device, class_names=classes)
