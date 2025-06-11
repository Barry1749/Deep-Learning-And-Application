import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
import timm

# ----------- EDIT THESE VALUES ----------------
model_name = "resmlp_24_224"
model_path = "/Users/billyhsieh/Desktop/DLA_Final_Project/resmlp_24_224.pth"
data_dir = "/Users/billyhsieh/Desktop/DLA_Final_Project/new_barry_data"
batch_size = 16
# ----------------------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ImageNet statistics
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Transform for test data
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
])

# Load test dataset
test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(test_dataset.classes)
print(f"Loaded {len(test_dataset)} test samples.")
print(f"Detected {num_classes} classes.")

# Load model
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"Loaded model '{model_name}' from '{model_path}'.")

def evaluate(model, loader):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    micro_accuracy = accuracy_score(all_labels, all_preds)
    macro_accuracy = balanced_accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    return micro_accuracy, macro_accuracy, macro_precision, micro_precision, macro_recall, micro_recall

micro_acc, macro_acc, macro_prec, micro_prec, macro_rec, micro_rec = evaluate(model, test_loader)

print("\n📊 Evaluation Results:")
print(f"  - Micro Accuracy   : {micro_acc:.4f}")
print(f"  - Macro Accuracy   : {macro_acc:.4f}")
print(f"  - Macro Precision  : {macro_prec:.4f}")
print(f"  - Micro Precision  : {micro_prec:.4f}")
print(f"  - Macro Recall     : {macro_rec:.4f}")
print(f"  - Micro Recall     : {micro_rec:.4f}")

# import os
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
# import timm

# # ----------- EDIT THESE VALUES ----------------
# model_name = "mobilevitv2_150.cvnets_in22k_ft_in1k"
# model_path = "/Users/billyhsieh/Desktop/DLA_Final_Project/best_model.pth"
# data_dir = "/Users/billyhsieh/Desktop/DLA_Final_Project/classified_data/test"
# batch_size = 16
# num_area_classes = 6  # 你固定設定的地區分類數
# # ----------------------------------------------

# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # ImageNet statistics
# imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# # Transform for test data
# test_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
# ])

# # Load test dataset
# test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# num_node_classes = len(test_dataset.classes)
# print(f"Loaded {len(test_dataset)} test samples.")
# print(f"Detected {num_node_classes} node classes.")

# # --------- Define MultiTaskModel (must match training script) ---------
# class MultiTaskModel(nn.Module):
#     def __init__(self, model_name, num_node_classes, num_area_classes):
#         super(MultiTaskModel, self).__init__()
#         self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
#         in_features = self.backbone.num_features
#         self.node_head = nn.Linear(in_features, num_node_classes)
#         self.area_head = nn.Linear(in_features, num_area_classes)

#     def forward(self, x):
#         features = self.backbone(x)
#         node_out = self.node_head(features)
#         area_out = self.area_head(features)
#         return node_out, area_out

# # Load model
# model = MultiTaskModel(model_name, num_node_classes=num_node_classes, num_area_classes=num_area_classes)
# state_dict = torch.load(model_path, map_location=device)
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()
# print(f"Loaded multitask model '{model_name}' from '{model_path}'.")

# # You should define this mapping (used if you want to evaluate area)
# # For demo: let's assign each node class to a dummy area (e.g., 0~5 repeat)
# node_to_area = torch.tensor([i % num_area_classes for i in range(num_node_classes)]).to(device)

# def evaluate(model, loader):
#     all_node_preds, all_node_labels = [], []
#     all_area_preds, all_area_labels = [], []

#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)

#             node_outputs, area_outputs = model(images)
#             node_preds = torch.argmax(node_outputs, dim=1)

#             # For area, use predicted node → corresponding area
#             area_labels = node_to_area[labels]
#             area_preds = node_to_area[node_preds]

#             all_node_preds.extend(node_preds.cpu().numpy())
#             all_node_labels.extend(labels.cpu().numpy())
#             all_area_preds.extend(area_preds.cpu().numpy())
#             all_area_labels.extend(area_labels.cpu().numpy())

#     # Node metrics
#     node_micro_acc = accuracy_score(all_node_labels, all_node_preds)
#     node_macro_acc = balanced_accuracy_score(all_node_labels, all_node_preds)
#     node_macro_prec = precision_score(all_node_labels, all_node_preds, average='macro', zero_division=0)
#     node_micro_prec = precision_score(all_node_labels, all_node_preds, average='micro', zero_division=0)
#     node_macro_rec = recall_score(all_node_labels, all_node_preds, average='macro', zero_division=0)
#     node_micro_rec = recall_score(all_node_labels, all_node_preds, average='micro', zero_division=0)

#     # Area metrics
#     area_micro_acc = accuracy_score(all_area_labels, all_area_preds)
#     area_macro_acc = balanced_accuracy_score(all_area_labels, all_area_preds)

#     return {
#         "node_micro_acc": node_micro_acc,
#         "node_macro_acc": node_macro_acc,
#         "node_macro_prec": node_macro_prec,
#         "node_micro_prec": node_micro_prec,
#         "node_macro_rec": node_macro_rec,
#         "node_micro_rec": node_micro_rec,
#         "area_micro_acc": area_micro_acc,
#         "area_macro_acc": area_macro_acc
#     }

# metrics = evaluate(model, test_loader)

# # Print results
# print("\n📊 Evaluation Results:")
# print("🧠 Node Classification:")
# print(f"  - Micro Accuracy   : {metrics['node_micro_acc']:.4f}")
# print(f"  - Macro Accuracy   : {metrics['node_macro_acc']:.4f}")
# print(f"  - Macro Precision  : {metrics['node_macro_prec']:.4f}")
# print(f"  - Micro Precision  : {metrics['node_micro_prec']:.4f}")
# print(f"  - Macro Recall     : {metrics['node_macro_rec']:.4f}")
# print(f"  - Micro Recall     : {metrics['node_micro_rec']:.4f}")
# print("📍 Area Classification (via node-to-area mapping):")
# print(f"  - Micro Accuracy   : {metrics['area_micro_acc']:.4f}")
# print(f"  - Macro Accuracy   : {metrics['area_macro_acc']:.4f}")
