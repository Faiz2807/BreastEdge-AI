#!/usr/bin/env python3
"""
BreastEdge AI - Kaggle Notebook
Full pipeline: Training, Evaluation, Inference

Author: Driss Faiz Ferhat
Competition: Google Edge AI Prize Track
"""

# ============================================================================
# SECTION 1: SETUP & IMPORTS
# ============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from monai.networks.nets import ResNet
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("BREASTEDGE AI - KAGGLE NOTEBOOK")
print("=" * 70)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# SECTION 2: DATASET PREPARATION
# ============================================================================

class BreastDataset(Dataset):
    """Breast histopathology dataset."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Example: Load dataset (adjust paths for your environment)
# data_dir = "/path/to/breast_data"
# benign_images = [(path, 0) for path in glob.glob(f"{data_dir}/*/0/*.png")]
# malignant_images = [(path, 1) for path in glob.glob(f"{data_dir}/*/1/*.png")]

# ============================================================================
# SECTION 3: MODEL DEFINITION
# ============================================================================

def create_resnet50(num_classes=2):
    """Create ResNet50 classifier."""
    model = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=2,
        n_input_channels=3,
        num_classes=num_classes
    )
    return model

print("\n📦 Model: ResNet50")
print(f"   Parameters: {sum(p.numel() for p in create_resnet50().parameters()):,}")

# ============================================================================
# SECTION 4: TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.0001):
    """Train ResNet50 classifier."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total * 100
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    return history, best_val_acc

# ============================================================================
# SECTION 5: EVALUATION
# ============================================================================

def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = (all_preds == all_labels).mean() * 100
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"Sensitivity: {sensitivity:.2f}%")
    print(f"Specificity: {specificity:.2f}%")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

# ============================================================================
# SECTION 6: INFERENCE WITH GOOGLE HAI-DEF MODELS
# ============================================================================

def load_hai_def_models():
    """Load Google HAI-DEF models for inference."""
    print("\n📥 Loading Google HAI-DEF models...")
    
    # MedSigLIP-448
    print("  MedSigLIP-448...", end="", flush=True)
    medsiglip_processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    medsiglip_model = AutoModel.from_pretrained(
        "google/medsiglip-448",
        torch_dtype=torch.float16
    ).to(device)
    medsiglip_model.eval()
    print(" ✓")
    
    # ResNet50
    print("  ResNet50...", end="", flush=True)
    classifier = create_resnet50().to(device)
    classifier.load_state_dict(torch.load("best_model.pth"))
    classifier.eval()
    print(" ✓")
    
    # MedGemma 1.5
    print("  MedGemma 1.5...", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-1.5-4b-it")
    medgemma = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    medgemma.eval()
    print(" ✓\n")
    
    return {
        "medsiglip_processor": medsiglip_processor,
        "medsiglip": medsiglip_model,
        "classifier": classifier,
        "tokenizer": tokenizer,
        "medgemma": medgemma
    }

def infer_with_explanation(image_path, models):
    """Full inference pipeline with explanation."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Extract MedSigLIP features
    inputs = models["medsiglip_processor"](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = models["medsiglip"].vision_model(**inputs).pooler_output
    
    # Classify with ResNet50
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = models["classifier"](img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    prediction = {
        "class": int(pred_class.item()),
        "confidence": float(confidence.item()),
        "prob_benign": float(probs[0, 0].item()),
        "prob_malignant": float(probs[0, 1].item())
    }
    
    # Generate MedGemma explanation
    pred_name = "malignant" if prediction["class"] == 1 else "benign"
    conf_pct = prediction["confidence"] * 100
    
    prompt = f"""Medical Education:

Histopathology: {pred_name.upper()} tissue ({conf_pct:.1f}% confidence)

Provide 2-sentence clinical explanation of {pred_name} breast tissue."""

    inputs = models["tokenizer"](prompt, return_tensors="pt").to(models["medgemma"].device)
    
    with torch.no_grad():
        outputs = models["medgemma"].generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=models["tokenizer"].eos_token_id
        )
    
    explanation = models["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    if prompt in explanation:
        explanation = explanation.split(prompt)[-1].strip()
    
    prediction["explanation"] = explanation[:200]
    
    return prediction

# ============================================================================
# SECTION 7: EXAMPLE USAGE
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE USAGE")
print("=" * 70)

print("""
# Step 1: Prepare dataset
# data = prepare_dataset("/path/to/breast_data")

# Step 2: Create model
model = create_resnet50().to(device)

# Step 3: Train
# history, best_acc = train_model(model, train_loader, val_loader, num_epochs=5)

# Step 4: Evaluate
# metrics = evaluate_model(model, test_loader)

# Step 5: Inference with HAI-DEF
# models = load_hai_def_models()
# result = infer_with_explanation("test_image.png", models)
# print(f"Prediction: {result['class']}")
# print(f"Confidence: {result['confidence']:.2f}")
# print(f"Explanation: {result['explanation']}")
""")

print("\n✅ Notebook Ready")
print("\nFor full execution, uncomment and run the example usage steps.")
