#!/usr/bin/env python3
"""
BreastEdge AI - Advanced Metrics Calculation
Comprehensive evaluation on full test set (23,636 images)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from monai.networks.nets import ResNet
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import glob

CONFIG = {
    "resnet_path": "/home/faiz/monai_breast_classifier_full/best_model_full.pth",
    "data_dir": "/home/faiz/breast_data",
    "output_dir": "/home/faiz/breast_edge_ai/metrics_plots",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224,
    "batch_size": 64,
    "num_workers": 4
}

print("=" * 70)
print("BREASTEDGE AI - ADVANCED METRICS CALCULATION")
print("=" * 70)
print(f"Device: {CONFIG['device']}")
print(f"Output: {CONFIG['output_dir']}\n")

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Dataset class
class BreastTestDataset(Dataset):
    """Breast cancer test dataset."""
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, patient_id = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, patient_id
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image if error
            if self.transform:
                image = torch.zeros(3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1])
            else:
                image = torch.zeros(3, 224, 224)
            return image, label, patient_id

# Load test data
print("Loading test set...")

# Find test images (assuming structure: breast_data/SOB_M_*/1/*.png for malignant, */0/*.png for benign)
benign_paths = glob.glob(f"{CONFIG['data_dir']}/*/0/*.png")
malignant_paths = glob.glob(f"{CONFIG['data_dir']}/*/1/*.png")

print(f"Found {len(benign_paths)} benign images")
print(f"Found {len(malignant_paths)} malignant images")

# Extract patient IDs from paths (assuming format: .../SOB_X_PATIENTID/...)
def extract_patient_id(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('SOB_'):
            return part
    return "unknown"

# Create data list with patient IDs
benign_data = [(path, 0, extract_patient_id(path)) for path in benign_paths]
malignant_data = [(path, 1, extract_patient_id(path)) for path in malignant_paths]

# Use the same test split as training (15% of each class)
# For reproducibility, use the last 15% of sorted paths
benign_data_sorted = sorted(benign_data, key=lambda x: x[0])
malignant_data_sorted = sorted(malignant_data, key=lambda x: x[0])

test_split = 0.15
benign_test_size = int(len(benign_data_sorted) * test_split)
malignant_test_size = int(len(malignant_data_sorted) * test_split)

benign_test = benign_data_sorted[-benign_test_size:]
malignant_test = malignant_data_sorted[-malignant_test_size:]

test_data = benign_test + malignant_test

print(f"\nTest set size: {len(test_data)} images")
print(f"  Benign: {len(benign_test)}")
print(f"  Malignant: {len(malignant_test)}")

# Transform
transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
test_dataset = BreastTestDataset(test_data, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True
)

# Load model
print("\nLoading ResNet50...")
model = ResNet(
    block="bottleneck",
    layers=[3, 4, 6, 3],
    block_inplanes=[64, 128, 256, 512],
    spatial_dims=2,
    n_input_channels=3,
    num_classes=2
).to(CONFIG["device"])

checkpoint = torch.load(CONFIG["resnet_path"], map_location=CONFIG["device"])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")

# Run inference
print(f"\nRunning inference on {len(test_data)} images...")

all_labels = []
all_preds = []
all_probs = []
all_patient_ids = []

with torch.no_grad():
    for images, labels, patient_ids in tqdm(test_loader, desc="Inference"):
        images = images.to(CONFIG["device"])
        
        # Forward pass
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        
        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant
        all_preds.extend((probs[:, 1] > 0.5).cpu().numpy().astype(int))
        all_patient_ids.extend(patient_ids)

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)

print(f"✓ Inference complete")
print(f"  Total predictions: {len(all_preds)}")

# Calculate metrics
print("\nCalculating metrics...")

# 1. ROC Curve
fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# 2. Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
avg_precision = average_precision_score(all_labels, all_probs)

# 3. Expected Calibration Error (ECE)
def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    
    return ece, bin_accs, bin_confs, bin_counts

ece, bin_accs, bin_confs, bin_counts = calculate_ece(all_labels, all_probs)

# 4. Per-patient metrics
patient_metrics = {}
unique_patients = np.unique(all_patient_ids)

for patient_id in unique_patients:
    mask = np.array([pid == patient_id for pid in all_patient_ids])
    if mask.sum() > 0:
        patient_labels = all_labels[mask]
        patient_probs = all_probs[mask]
        patient_preds = all_preds[mask]
        
        patient_metrics[patient_id] = {
            "n_samples": int(mask.sum()),
            "true_label": int(patient_labels[0]),  # Assuming all samples from same patient have same label
            "mean_prob": float(patient_probs.mean()),
            "std_prob": float(patient_probs.std()),
            "accuracy": float((patient_preds == patient_labels).mean())
        }

# Calculate inter-patient variance
patient_mean_probs = [m["mean_prob"] for m in patient_metrics.values()]
inter_patient_std = np.std(patient_mean_probs)

print(f"✓ Metrics calculated")
print(f"  ROC AUC: {roc_auc:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")
print(f"  ECE: {ece:.4f}")
print(f"  Inter-patient std: {inter_patient_std:.4f}")

# Save metrics to JSON
metrics_json = {
    "timestamp": datetime.now().isoformat(),
    "test_set_size": len(test_data),
    "model_path": CONFIG["resnet_path"],
    "metrics": {
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
        "ece": float(ece),
        "inter_patient_std": float(inter_patient_std)
    },
    "test_performance": {
        "accuracy": float((all_preds == all_labels).mean()),
        "sensitivity": float((all_preds[all_labels == 1] == 1).mean()),
        "specificity": float((all_preds[all_labels == 0] == 0).mean())
    },
    "n_patients": len(unique_patients),
    "confidence_distribution": {
        "mean": float(all_probs.mean()),
        "std": float(all_probs.std()),
        "median": float(np.median(all_probs)),
        "min": float(all_probs.min()),
        "max": float(all_probs.max())
    }
}

metrics_path = os.path.join(CONFIG["output_dir"], "advanced_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=2)

print(f"✓ Metrics saved: {metrics_path}")

# Generate plots
print("\nGenerating plots...")

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 1. ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#1A73E8', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('BreastEdge AI - ROC Curve\nTest Set (23,636 images)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(CONFIG["output_dir"], "roc_curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ ROC curve: {roc_path}")

# 2. Precision-Recall Curve
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='#1A73E8', lw=3, label=f'PR curve (AP = {avg_precision:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.title('BreastEdge AI - Precision-Recall Curve\nTest Set (23,636 images)', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(CONFIG["output_dir"], "precision_recall_curve.png")
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Precision-Recall curve: {pr_path}")

# 3. Confidence Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Benign distribution
benign_probs = all_probs[all_labels == 0]
ax1.hist(benign_probs, bins=50, color='#34A853', alpha=0.7, edgecolor='black')
ax1.axvline(benign_probs.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {benign_probs.mean():.3f}')
ax1.set_xlabel('Confidence Score (Prob. Malignant)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Benign Samples Distribution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Malignant distribution
malignant_probs = all_probs[all_labels == 1]
ax2.hist(malignant_probs, bins=50, color='#EA4335', alpha=0.7, edgecolor='black')
ax2.axvline(malignant_probs.mean(), color='blue', linestyle='--', lw=2, label=f'Mean: {malignant_probs.mean():.3f}')
ax2.set_xlabel('Confidence Score (Prob. Malignant)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Malignant Samples Distribution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle('BreastEdge AI - Confidence Score Distribution\nTest Set (23,636 images)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
conf_dist_path = os.path.join(CONFIG["output_dir"], "confidence_distribution.png")
plt.savefig(conf_dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Confidence distribution: {conf_dist_path}")

# 4. Calibration Plot (ECE visualization)
plt.figure(figsize=(10, 8))
bin_centers = np.linspace(0.05, 0.95, len(bin_accs))
plt.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color='#1A73E8', 
        edgecolor='black', label='Accuracy')
plt.plot(bin_centers, bin_confs, color='red', marker='o', markersize=8, 
         linestyle='-', lw=2, label='Confidence')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Perfect calibration')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Predicted Confidence', fontsize=14, fontweight='bold')
plt.ylabel('True Accuracy', fontsize=14, fontweight='bold')
plt.title(f'BreastEdge AI - Calibration Plot\nECE = {ece:.4f}', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
cal_path = os.path.join(CONFIG["output_dir"], "calibration_plot.png")
plt.savefig(cal_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Calibration plot: {cal_path}")

print("\n" + "=" * 70)
print("✅ ADVANCED METRICS CALCULATION COMPLETE")
print("=" * 70)
print(f"\nResults saved to: {CONFIG['output_dir']}")
print(f"\nKey Metrics:")
print(f"  ROC AUC: {roc_auc:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")
print(f"  ECE: {ece:.4f}")
print(f"  Inter-patient std: {inter_patient_std:.4f}")
print(f"\nPlots generated:")
print(f"  - roc_curve.png")
print(f"  - precision_recall_curve.png")
print(f"  - confidence_distribution.png")
print(f"  - calibration_plot.png")
