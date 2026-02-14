#!/usr/bin/env python3
"""
Select 5 diverse demo images from the breast dataset
- 2 benign (high confidence)
- 2 malignant (high confidence)
- 1 borderline (medium confidence ~60-70%)
"""

import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from monai.networks.nets import ResNet

# Load model
CONFIG = {
    "resnet_path": "/home/faiz/monai_breast_classifier_full/best_model_full.pth",
    "data_dir": "/home/faiz/breast_data",
    "output_dir": "/home/faiz/breast_edge_ai/demo_images",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print("Loading ResNet50...")
classifier = ResNet(
    block="bottleneck",
    layers=[3, 4, 6, 3],
    block_inplanes=[64, 128, 256, 512],
    spatial_dims=2,
    n_input_channels=3,
    num_classes=2
).to(CONFIG["device"])

checkpoint = torch.load(CONFIG["resnet_path"], map_location=CONFIG["device"])
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()
print("✓ Model loaded")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(CONFIG["device"])
    
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    
    return {
        "class": int(pred_class.item()),
        "confidence": float(confidence.item()),
        "prob_benign": float(probs[0, 0].item()),
        "prob_malignant": float(probs[0, 1].item())
    }

# Scan dataset
print("\nScanning dataset...")
data_path = Path(CONFIG["data_dir"])
benign_high = []
malignant_high = []
borderline = []

count = 0
for patient_dir in list(data_path.iterdir())[:30]:  # Sample first 30 patients
    if not patient_dir.is_dir():
        continue
    
    # Benign
    class0_dir = patient_dir / "0"
    if class0_dir.exists():
        for img_path in list(class0_dir.glob("*.png"))[:5]:
            pred = predict_image(img_path)
            if pred["class"] == 0 and pred["confidence"] > 0.95:
                benign_high.append((str(img_path), pred))
            elif 0.6 <= pred["confidence"] <= 0.75:
                borderline.append((str(img_path), pred))
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count} images...")
    
    # Malignant
    class1_dir = patient_dir / "1"
    if class1_dir.exists():
        for img_path in list(class1_dir.glob("*.png"))[:5]:
            pred = predict_image(img_path)
            if pred["class"] == 1 and pred["confidence"] > 0.95:
                malignant_high.append((str(img_path), pred))
            elif 0.6 <= pred["confidence"] <= 0.75:
                borderline.append((str(img_path), pred))
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count} images...")

print(f"\n✓ Found:")
print(f"  Benign high conf: {len(benign_high)}")
print(f"  Malignant high conf: {len(malignant_high)}")
print(f"  Borderline: {len(borderline)}")

# Select images
selected = []

if len(benign_high) >= 2:
    selected.extend(random.sample(benign_high, 2))
    print("\n✓ Selected 2 benign (high confidence)")
else:
    print(f"⚠️  Only {len(benign_high)} benign high confidence images found")

if len(malignant_high) >= 2:
    selected.extend(random.sample(malignant_high, 2))
    print("✓ Selected 2 malignant (high confidence)")
else:
    print(f"⚠️  Only {len(malignant_high)} malignant high confidence images found")

if len(borderline) >= 1:
    selected.append(random.choice(borderline))
    print("✓ Selected 1 borderline case")
else:
    print("⚠️  No borderline cases found")

# Copy files
os.makedirs(CONFIG["output_dir"], exist_ok=True)

names = [
    "demo_benign_1.png",
    "demo_benign_2.png",
    "demo_malignant_1.png",
    "demo_malignant_2.png",
    "demo_borderline.png"
]

print("\n📋 Demo images:")
for i, ((img_path, pred), name) in enumerate(zip(selected, names)):
    dest = os.path.join(CONFIG["output_dir"], name)
    shutil.copy(img_path, dest)
    print(f"  {name}:")
    print(f"    Class: {'MALIGNANT' if pred['class'] == 1 else 'BENIGN'}")
    print(f"    Confidence: {pred['confidence']*100:.1f}%")
    print(f"    Source: {img_path}")

print(f"\n✅ Demo images saved to {CONFIG['output_dir']}")
