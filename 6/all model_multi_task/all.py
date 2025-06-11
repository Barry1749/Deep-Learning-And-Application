# This is the completed version of your training script with multitasking support.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
import datetime

# =====================
# Parameters and setup
# =====================
num_epochs = 20
early_stop_patience = 4
loss_alpha = 1.0
loss_beta = 0.5
batch_size = 16
log_file = "model_results.txt"

base_dir = "/Users/billyhsieh/Desktop/DLA_Final_Project/classified_data"

imagenet_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
])

train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(base_dir, 'validation'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("類別數量：", len(train_dataset.classes))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# =====================
# Model Definition
# =====================
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_node_classes, num_area_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # Remove head
        self.node_head = nn.Linear(self.backbone.num_features, num_node_classes)
        self.area_head = nn.Linear(self.backbone.num_features, num_area_classes)

    def forward(self, x):
        features = self.backbone(x)
        node_out = self.node_head(features)
        area_out = self.area_head(features)
        return node_out, area_out

# =====================
# Custom Loss Function
# =====================

# -------------------------
# customed loss function
# ----- 1. Node-to-Node Distance Matrix -----
node_distance_matrix = torch.tensor([
    [0,1,2,2,1,2,2,2.5,3.5,4,5.5,5,4,2.5,3.5,4,5,3.5,4.5,3.5,2.5,2.5],
    [1,0,1,3,2,3,1,1.5,2.5,3,4.5,4,3,1.5,4.5,5,6,4.5,5.5,4.5,3.5,3.5],
    [2,1,0,2,3,4,2,2.5,3.5,4,5.5,3,2,0.5,3.5,4,5,3.5,4.5,5.5,4.5,2.5],
    [2,3,2,0,1,2,4,4.5,5.5,6,7.5,5,4,2.5,1.5,2,3,1.5,2.5,3.5,2.5,0.5],
    [1,2,3,1,0,1,3,3.5,4.5,5,6.5,6,5,3.5,2.5,3,4,2.5,3.5,2.5,1.5,1.5],
    [2,3,4,2,1,0,2,2.5,3.5,4,5.5,7,6,4.5,3.5,4,5,3.5,2.5,1.5,0.5,2.5],
    [2,1,2,4,3,2,0,0.5,1.5,2,3.5,5,4,2.5,5.5,6,7,5.5,4.5,3.5,2.5,4.5],
    [2.5,1.5,2.5,4.5,3.5,2.5,0.5,0,1,1.5,3,3.5,4.5,3,6,6.5,7.5,6,5,4,3,5],
    [3.5,2.5,3.5,5.5,4.5,3.5,1.5,1,0,0.5,2,2.5,3.5,4,7,7.5,8.5,7,6,5,4,6],
    [4,3,4,6,5,4,2,1.5,0.5,0,1.5,2,3,4.5,7.5,8,9,7.5,6.5,5.5,4.5,6.5],
    [5.5,4.5,5.5,7.5,6.5,5.5,3.5,3,2,1.5,0,0.5,1.5,3,7,7.5,8.5,7,8,7,6,6.5],
    [5,4,3,5,6,7,5,3.5,2.5,2,0.5,0,1,2.5,6.5,7,8,6.5,7.5,7.5,6.5,5.5],
    [4,3,2,4,5,6,4,4.5,3.5,3,1.5,1,0,1.5,5.5,6,7,5.5,6.5,7.5,7,4.5],
    [2.5,1.5,0.5,2.5,3.5,4.5,2.5,3,4,4.5,3,2.5,1.5,0,4,4.5,5.5,4,5,6,5.5,3],
    [3.5,4.5,3.5,1.5,2.5,3.5,5.5,6,7,7.5,7,6.5,5.5,4,0,0.5,1.5,0.5,1.5,2.5,3.5,1],
    [4,5,4,2,3,4,6,6.5,7.5,8,7.5,7,6,4.5,0.5,0,1,1,2,3,4,1.5],
    [5,6,5,3,4,5,7,7.5,8.5,9,8.5,8,7,5.5,1.5,1,0,1.5,2.5,3.5,4.5,2.5],
    [3.5,4.5,3.5,1.5,2.5,3.5,5.5,6,7,7.5,7,6.5,5.5,4,0.5,1,1.5,0,1,2,3,1],
    [4.5,5.5,4.5,2.5,3.5,2.5,4.5,5,6,6.5,8,7.5,6.5,5,1.5,2,2.5,1,0,1,2,2],
    [3.5,4.5,5.5,3.5,2.5,1.5,3.5,4,5,5.5,7,7.5,7.5,6,2.5,3,3.5,2,1,0,1,3],
    [2.5,3.5,4.5,2.5,1.5,0.5,2.5,3,4,4.5,6,6.5,7,5.5,3.5,4,4.5,3,2,1,0,3],
    [2.5,3.5,2.5,0.5,1.5,2.5,4.5,5,6,6.5,6.5,5.5,4.5,3,1,1.5,2.5,1,2,3,3,0]
], dtype=torch.float)

# ----- 2. Area-to-Area Distance Matrix -----
area_distance_matrix = torch.tensor([
    [0, 1, 2, 2, 2, 2],
    [1, 0, 1, 1, 1, 1],
    [2, 1, 0, 1, 3, 3],
    [2, 1, 1, 0, 3, 3],
    [2, 1, 3, 3, 0, 1],
    [2, 1, 3, 3, 1, 0]
], dtype=torch.float)

# ----- 3. Node-to-Area Mapping -----
node_to_area = torch.tensor([
    0, 0, 0, 0, 0, 0, 0,    # 1~7 → area 0
    1,                      # 8  → area 1
    2, 2,                   # 9~10 → area 2
    3, 3, 3,                # 11~13 → area 3
    1,                      # 14 → area 1
    4, 4, 4,                # 15~17 → area 4
    5, 5, 5,                # 18~20 → area 5
    1,                      # 21 → area 1
    1                       # 22 → area 1
], dtype=torch.long)

class DistancePenaltyLoss(nn.Module):
    def __init__(self, node_dist_matrix, area_dist_matrix, node_to_area, alpha=1.0, beta=1.0):
        super().__init__()
        self.node_dist = node_dist_matrix.to(device)
        self.area_dist = area_dist_matrix.to(device)
        self.node_to_area = node_to_area.to(device)
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, node_pred, area_pred, node_target):
        area_target = self.node_to_area[node_target]
        ce_loss = self.ce(node_pred, node_target)
        ce_area_loss = self.ce(area_pred, area_target)

        pred_softmax = F.softmax(node_pred, dim=1)
        penalty = torch.sum(pred_softmax * self.node_dist[node_target], dim=1).mean()

        return self.alpha * ce_loss + self.beta * ce_area_loss + 0.01 * penalty

# =====================
# Training Loop
# =====================
def train(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device.type):
            node_pred, area_pred = model(images)
            loss = criterion(node_pred, area_pred, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(node_pred, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        batch_acc = correct / labels.size(0)
        print(f"Batch {batch_idx + 1}/{len(loader)} Accuracy: {batch_acc:.4f}")

    epoch_accuracy = total_correct / total_samples
    print(f"Train Epoch Accuracy: {epoch_accuracy:.4f}")

    return total_loss / len(loader.dataset)

# =====================
# Evaluation Loop
# =====================
def evaluate(model, loader):
    model.eval()
    node_preds, node_targets = [], []
    area_preds, area_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            node_out, area_out = model(images)
            node_preds.extend(torch.argmax(node_out, dim=1).cpu().numpy())
            area_preds.extend(torch.argmax(area_out, dim=1).cpu().numpy())
            node_targets.extend(labels.cpu().numpy())
            area_targets.extend(node_to_area[labels.cpu()].cpu().numpy())

    node_acc = accuracy_score(node_targets, node_preds)
    area_acc = accuracy_score(area_targets, area_preds)
    return node_acc, area_acc

# =====================
# Main Loop
# =====================
model = MultiTaskModel(
    model_name="mobilevitv2_150.cvnets_in22k_ft_in1k",
    num_node_classes=22,
    num_area_classes=6
).to(device)

criterion = DistancePenaltyLoss(
    node_distance_matrix, area_distance_matrix, node_to_area,
    alpha=loss_alpha, beta=loss_beta
)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler(enabled=torch.cuda.is_available())

best_val_acc = 0
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, scaler, criterion)
    val_node_acc, val_area_acc = evaluate(model, val_loader)

    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Node Acc: {val_node_acc:.4f}, Area Acc: {val_area_acc:.4f}\n")

    if val_node_acc > best_val_acc:
        best_val_acc = val_node_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

# =====================
# Final Evaluation
# =====================
model.load_state_dict(torch.load("best_model.pth"))
test_node_acc, test_area_acc = evaluate(model, test_loader)

print(f"Test Node Accuracy: {test_node_acc:.4f}, Test Area Accuracy: {test_area_acc:.4f}")