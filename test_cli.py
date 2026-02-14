#!/usr/bin/env python3
"""
BreastEdge AI - CLI Test
Test the full pipeline (MedSigLIP + ResNet50 + MedGemma) from command line
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from monai.networks.nets import ResNet
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor

CONFIG = {
    "resnet_path": "/home/faiz/monai_breast_classifier_full/best_model_full.pth",
    "medgemma_model": "google/medgemma-1.5-4b-it",
    "medsiglip_model": "google/medsiglip-448",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224
}

print("=" * 70)
print("BREASTEDGE AI - CLI TEST")
print("=" * 70)
print(f"Device: {CONFIG['device']}\n")

# Load models
print("Loading MedSigLIP-448...")
medsiglip_processor = AutoProcessor.from_pretrained(CONFIG["medsiglip_model"])
medsiglip_model = AutoModel.from_pretrained(
    CONFIG["medsiglip_model"],
    torch_dtype=torch.float16
).to(CONFIG["device"])
medsiglip_model.eval()
print("✓ MedSigLIP loaded")

print("\nLoading ResNet50...")
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
print(f"✓ ResNet50 loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")

print("\nLoading MedGemma 1.5...")
medgemma_tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["medgemma_model"],
    local_files_only=True
)
medgemma = AutoModelForCausalLM.from_pretrained(
    CONFIG["medgemma_model"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
medgemma.eval()
print("✓ MedGemma 1.5 loaded\n")

resnet_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract MedSigLIP features."""
    image = Image.open(image_path).convert("RGB")
    inputs = medsiglip_processor(images=image, return_tensors="pt").to(CONFIG["device"])
    with torch.no_grad():
        vision_outputs = medsiglip_model.vision_model(**inputs)
        features = vision_outputs.pooler_output
    return features

def classify(image_path):
    """Classify with ResNet50."""
    image = Image.open(image_path).convert("RGB")
    img_tensor = resnet_transform(image).unsqueeze(0).to(CONFIG["device"])
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

def explain(prediction):
    """Generate MedGemma explanation."""
    pred_class = "malignant" if prediction["class"] == 1 else "benign"
    confidence = prediction["confidence"] * 100
    
    prompt = f"""Medical Education:

Histopathology result: {pred_class.upper()} tissue ({confidence:.1f}% confidence)

Provide a 2-sentence clinical explanation of {pred_class} breast tissue features."""

    inputs = medgemma_tokenizer(prompt, return_tensors="pt").to(medgemma.device)
    
    with torch.no_grad():
        outputs = medgemma.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=medgemma_tokenizer.eos_token_id
        )
    
    response = medgemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if prompt in response:
        explanation = response.split(prompt)[-1].strip()
    else:
        explanation = response.strip()
    
    return explanation[:250]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_cli.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 70)
    print(f"Testing image: {image_path}")
    print("=" * 70)
    
    # Extract features
    print("\n1. Extracting MedSigLIP features...")
    features = extract_features(image_path)
    print(f"✓ Features extracted: shape {features.shape}")
    
    # Classify
    print("\n2. Classifying with ResNet50...")
    prediction = classify(image_path)
    pred_label = "🔴 MALIGNANT" if prediction["class"] == 1 else "🟢 BENIGN"
    print(f"✓ Prediction: {pred_label}")
    print(f"  Confidence: {prediction['confidence']*100:.1f}%")
    print(f"  P(Benign): {prediction['prob_benign']*100:.1f}%")
    print(f"  P(Malignant): {prediction['prob_malignant']*100:.1f}%")
    
    # Explain
    print("\n3. Generating MedGemma explanation...")
    explanation = explain(prediction)
    print(f"✓ Explanation:")
    print(f"  {explanation}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
