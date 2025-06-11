import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
import datetime

# Set up global values
num_epochs = 30
early_stop_patience = 4
loss_alpha = 1.0
loss_beta = 0.5
batch_size = 16
log_file = "model_results.txt"

# ---------------------
base_dir = "/Users/billyhsieh/Desktop/DLA_Final_Project/classified_data"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
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

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
])

train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(base_dir, 'validation'), transform=validation_transform)
test_data = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("類別數量：", len(train_dataset.classes))
# ---------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Extended list of models with pretrained weights in timm
model_names = [
    # Mobile & Efficient Models
    # "mobilenetv3_large_100.ra_in1k",
    # "mobilenetv2_100.ra_in1k",
    # "mobilevitv2_100.cvnets_in1k",
    "mobilevitv2_150.cvnets_in22k_ft_in1k",
    # "efficientnet_b0.ra_in1k",
    # "efficientvit_m0.r224_in1k",
    # "efficientformerv2_s1.snap_dist_in1k",
    # "mobileone_s1.apple_in1k",

    # # Standard + Strong CNNs
    # "resnet50.ra_in1k",
    # "resnet50d.ra2_in1k",
    # "res2net50d.in1k",
    # "resnet50.fb_swsl_ig1b_ft_in1k",

    # # ConvNeXt family
    # "convnext_tiny.fb_in1k",
    # "convnext_small.fb_in1k",
    # "convnext_base.fb_in1k",
    # "convnextv2_tiny.fcmae",

    # # Swin Transformer family
    # "swin_tiny_patch4_window7_224.ms_in1k",
    # "swinv2_tiny_window8_256.ms_in1k",
    # "swinv2_cr_tiny_ns_224.sw_in1k",

    # # Other ViT and Transformer variants
    # "vit_base_patch16_224.in21k_ft_in1k",
    # "beit_base_patch16_224.in22k_ft_in1k",
    # "deit_base_distilled_patch16_224.in1k",
    # "cait_m36_384",
]

# Ensure clean results log
with open(log_file, "w") as f:
    f.write(f"Model Evaluation Log - {datetime.datetime.now()}\n\n")

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

# ----- 4. Custom Loss Function -----
class DistancePenaltyLoss(nn.Module):
    def __init__(self, node_distance_matrix, area_distance_matrix, node_to_area, alpha=1.0, beta=1.0):
        super().__init__()
        self.node_distance_matrix = node_distance_matrix
        self.area_distance_matrix = area_distance_matrix
        self.node_to_area = node_to_area
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        logits: (batch_size, 22)
        targets: (batch_size,) - ground truth class indices
        """
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)  # (B, 22)
        batch_size = targets.size(0)

        node_penalty = 0.0
        area_penalty = 0.0

        for i in range(batch_size):
            target = targets[i]
            prob = probs[i]  # shape (22,)

            # Node distance penalty
            node_distances = self.node_distance_matrix[target]  # (22,)
            node_penalty += (prob * node_distances).sum()

            # Area distance penalty
            target_area = self.node_to_area[target]
            pred_areas = self.node_to_area  # shape (22,)
            area_dists = self.area_distance_matrix[target_area][pred_areas]  # shape (22,)
            area_penalty += (prob * area_dists).sum()

        node_penalty /= batch_size
        area_penalty /= batch_size

        total_loss = ce_loss + self.alpha * node_penalty + self.beta * area_penalty
        return total_loss
# -------------------------
class DistancePenaltyLoss(nn.Module):
    def __init__(self, node_distance_matrix, area_distance_matrix, node_to_area, alpha=1.0, beta=1.0):
        super().__init__()
        self.node_distance_matrix = node_distance_matrix
        self.area_distance_matrix = area_distance_matrix
        self.node_to_area = node_to_area
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        batch_size = targets.size(0)
        node_penalty, area_penalty = 0.0, 0.0
        for i in range(batch_size):
            target = targets[i]
            prob = probs[i]
            node_distances = self.node_distance_matrix[target]
            node_penalty += (prob * node_distances).sum()
            target_area = self.node_to_area[target]
            pred_areas = self.node_to_area
            area_dists = self.area_distance_matrix[target_area][pred_areas]
            area_penalty += (prob * area_dists).sum()
        node_penalty /= batch_size
        area_penalty /= batch_size
        total_loss = ce_loss + self.alpha * node_penalty + self.beta * area_penalty
        return total_loss

# -------------------------
# Training and Evaluation
# -------------------------
def train_one_epoch_amp(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if device.type == "cuda":
            with torch.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # For mps or cpu: no AMP
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_batch = (predicted == labels).sum().item()
        total += labels.size(0)
        correct += correct_batch

        batch_acc = correct_batch / labels.size(0)
        print(f"Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.4f}")

    epoch_loss = total_loss / len(loader)
    epoch_acc = correct / total
    print(f"Epoch Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    micro = accuracy_score(all_labels, all_preds)
    macro = balanced_accuracy_score(all_labels, all_preds)
    return micro, macro

# -------------------------
# Main Loop for All Models
# -------------------------
for model_name in model_names:
    print(f"\n🚀 Training model: {model_name}")
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=len(train_loader.dataset.classes)).to(device)

        criterion = DistancePenaltyLoss(
            node_distance_matrix.to(device),
            area_distance_matrix.to(device),
            node_to_area.to(device),
            alpha=loss_alpha,
            beta=loss_beta
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        if torch.cuda.is_available():
            scaler = GradScaler()
        elif torch.backends.mps.is_available():
            # MPS does not support AMP currently
            scaler = None
        else:
            scaler = None

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = train_one_epoch_amp(model, train_loader, optimizer, criterion, scaler)
            val_micro_acc, val_macro_acc = evaluate(model, val_loader)
            print(f"📊 Val Micro: {val_micro_acc:.4f}, Macro: {val_macro_acc:.4f}")

            if val_micro_acc > best_val_acc:
                best_val_acc = val_micro_acc
                patience_counter = 0
                torch.save(model.state_dict(), f"{model_name}.pth")
                print(f"✅ Best model saved: {model_name}.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("🛑 Early stopping triggered.")
                    break

        # Load best model for testing
        model.load_state_dict(torch.load(f"{model_name}.pth"))
        model.to(device)
        test_micro_acc, test_macro_acc = evaluate(model, test_loader)

        with open(log_file, "a") as f:
            f.write(f"{model_name}:\n")
            f.write(f"  Test Micro Accuracy: {test_micro_acc:.4f}\n")
            f.write(f"  Test Macro Accuracy: {test_macro_acc:.4f}\n\n")
        print(f"📝 Logged performance of {model_name}")

    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"{model_name}:\n  ❌ Error occurred: {str(e)}\n\n")
        print(f"❌ Failed on model {model_name}: {e}")
