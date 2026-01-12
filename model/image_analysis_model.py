import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn as nn
import numpy as np

class MultiCategoryFoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Assign numeric labels for categories
        categories = sorted(os.listdir(root_dir))
        for idx, category in enumerate(categories):
            self.class_to_idx[category] = idx

            img_dir = os.path.join(root_dir, category, "images")
            label_dir = os.path.join(root_dir, category, "labels")

            for img_name in os.listdir(img_dir):
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(img_dir, img_name)
                    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".json"))
                    if os.path.exists(label_path):
                        self.samples.append((img_path, label_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, class_idx = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load JSON label
        with open(label_path, "r") as f:
            data = json.load(f)

        # Example: Use Nutri-score as label (A–E)
        nutri_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        nutri_score = data.get("nutri_score", "c").upper()
        nutri_label = nutri_map.get(nutri_score, 2)

        # You can also return both category and nutrition label if needed
        return image, class_idx, nutri_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = os.path.join("train")
test_dir = os.path.join("test")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3

# Display a sample
print(f"Classes: {train_dataset.class_to_idx}")
# img, category, nutri = dataset[0]
# print("Category ID:", category)
# print("Nutri-score label:", nutri)
# print("Image Tensor Shape:", img.shape)

# Use the default pretrained weights

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.class_to_idx))  # categories

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
# History for plotting
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    train_loss = total_loss / total_samples if total_samples > 0 else 0
    train_acc = total_correct / total_samples if total_samples > 0 else 0

    # Validation / Test evaluation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = images.size(0)
            val_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_samples += batch_size

    val_loss = val_loss / val_samples if val_samples > 0 else 0
    val_acc = val_correct / val_samples if val_samples > 0 else 0

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Record history for plotting
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

MODEL_PATH = "food_resnet18.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")

# Plot training curves (loss & accuracy)
# Summary: print min/max for train and test (validation) loss and accuracy
if train_loss_history:
    print(f"Train Loss: min {min(train_loss_history):.4f} | max {max(train_loss_history):.4f}")
else:
    print("Train Loss: no history available")

if val_loss_history:
    print(f"Test Loss: min {min(val_loss_history):.4f} | max {max(val_loss_history):.4f}")
else:
    print("Test Loss: no history available")

if train_acc_history:
    print(f"Train Acc : min {min(train_acc_history):.4f} | max {max(train_acc_history):.4f}")
else:
    print("Train Acc: no history available")

if val_acc_history:
    print(f"Test Acc : min {min(val_acc_history):.4f} | max {max(val_acc_history):.4f}")
else:
    print("Test Acc: no history available")

try:
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_loss_history) + 1))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Train Acc')
    plt.plot(epochs, val_acc_history, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    fig_path = 'training_curves.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Training curves saved to {fig_path}")
except ImportError:
    print('matplotlib not installed; skipping plot generation')

# Compute F1 score, confusion matrix and ROC curve (saved as images)
try:
    from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    # Collect predictions and probabilities on the test set
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(all_probs) == 0:
        print('No test predictions collected; skipping sklearn metrics')
    else:
        all_probs = np.vstack(all_probs)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # Weighted F1
        try:
            f1 = f1_score(y_true, y_pred, average='weighted')
            print(f"Weighted F1 (test): {f1:.4f}")
        except Exception as e:
            print('Could not compute F1 score:', e)

        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion matrix')
            plt.colorbar()
            classes = getattr(train_dataset, 'classes', None) or list(range(cm.shape[0]))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            cm_path = 'confusion_matrix.png'
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix saved to {cm_path}")
        except Exception as e:
            print('Could not compute/save confusion matrix:', e)

        # Multiclass ROC (one-vs-rest)
        try:
            n_classes = all_probs.shape[1]
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            plt.figure(figsize=(8, 6))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                label = f"{classes[i]} (AUC = {roc_auc:.2f})" if i < len(classes) else f"class {i} (AUC = {roc_auc:.2f})"
                plt.plot(fpr, tpr, lw=2, label=label)
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multiclass ROC')
            plt.legend(loc='lower right', fontsize='small')
            roc_path = 'roc_curve.png'
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC curve saved to {roc_path}")
        except Exception as e:
            print('Could not compute/save ROC curves:', e)
except ImportError:
    print('sklearn not installed; skipping F1/confusion/ROC generation')


# --- Helpers for single-image inference (real-world testing) ---
def load_model_checkpoint(model, checkpoint_path, device=None):
    """Load model state_dict from `checkpoint_path` onto `device` and return model.

    Args:
        model: torch.nn.Module instance (architecture must match checkpoint).
        checkpoint_path: path to .pth state_dict file.
        device: torch.device or None (auto-selects CUDA if available).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, transform, class_names=None, device=None, topk=3):
    """Run a single image through `model` and return top-k predictions with probabilities.

    Returns a list of tuples: [(class_name_or_idx, prob), ...] sorted by prob desc.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    if transform is not None:
        tensor = transform(img).unsqueeze(0).to(device)
    else:
        # Minimal conversion if no transform provided
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idx = probs.topk(topk, dim=1)

    top_probs = top_probs.cpu().numpy().squeeze()
    top_idx = top_idx.cpu().numpy().squeeze()

    if topk == 1:
        top_probs = [float(top_probs)]
        top_idx = [int(top_idx)]

    if class_names is not None:
        names = [class_names[i] for i in top_idx]
    else:
        names = [int(i) for i in top_idx]

    return list(zip(names, [float(p) for p in top_probs]))


# Example usage (commented):
# model = models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(train_dataset.class_to_idx))
# model = load_model_checkpoint(model, 'food_resnet18.pth')
# image_path = 'path/to/real_image.jpg'
# preds = predict_image(model, image_path, transform, class_names=getattr(train_dataset, 'classes', None), topk=3)
# print('Top predictions:', preds)
