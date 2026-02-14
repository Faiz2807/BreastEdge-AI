#!/usr/bin/env python3
"""
BreastEdge AI - Test All Demo Images
Batch test 5 demo images and save outputs
"""

import os
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
    "demo_images_dir": "/home/faiz/breast_edge_ai/demo_images",
    "output_dir": "/home/faiz/breast_edge_ai/demo_outputs",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224
}

print("=" * 70)
print("BREASTEDGE AI - BATCH TEST ALL DEMO IMAGES")
print("=" * 70)
print(f"Device: {CONFIG['device']}\n")

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Load models
print("Loading models...")
print("  MedSigLIP-448...", end="", flush=True)
medsiglip_processor = AutoProcessor.from_pretrained(CONFIG["medsiglip_model"])
medsiglip_model = AutoModel.from_pretrained(
    CONFIG["medsiglip_model"],
    torch_dtype=torch.float16
).to(CONFIG["device"])
medsiglip_model.eval()
print(" ✓")

print("  ResNet50...", end="", flush=True)
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
print(f" ✓ (Val Acc: {checkpoint['val_acc']:.2f}%)")

print("  MedGemma 1.5...", end="", flush=True)
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
print(" ✓\n")

resnet_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_path):
    """Process image through full pipeline."""
    # Extract features
    image = Image.open(image_path).convert("RGB")
    inputs = medsiglip_processor(images=image, return_tensors="pt").to(CONFIG["device"])
    with torch.no_grad():
        vision_outputs = medsiglip_model.vision_model(**inputs)
        features = vision_outputs.pooler_output
    
    # Classify
    img_tensor = resnet_transform(image).unsqueeze(0).to(CONFIG["device"])
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    
    prediction = {
        "class": int(pred_class.item()),
        "confidence": float(confidence.item()),
        "prob_benign": float(probs[0, 0].item()),
        "prob_malignant": float(probs[0, 1].item())
    }
    
    # Explain
    pred_class_name = "malignant" if prediction["class"] == 1 else "benign"
    conf_pct = prediction["confidence"] * 100
    
    prompt = f"""Medical Education:

Histopathology result: {pred_class_name.upper()} tissue ({conf_pct:.1f}% confidence)

Provide a 2-sentence clinical explanation of {pred_class_name} breast tissue features."""

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
    
    prediction["explanation"] = explanation[:250]
    prediction["features_shape"] = str(features.shape)
    
    return prediction

# Test all demo images
demo_images = [
    "demo_benign_1.png",
    "demo_benign_2.png",
    "demo_malignant_1.png",
    "demo_malignant_2.png",
    "demo_borderline.png"
]

print("=" * 70)
print("TESTING DEMO IMAGES")
print("=" * 70)

results = []

for i, img_name in enumerate(demo_images, 1):
    img_path = os.path.join(CONFIG["demo_images_dir"], img_name)
    
    if not os.path.exists(img_path):
        print(f"\n{i}/5 ⚠️  {img_name} - NOT FOUND")
        continue
    
    print(f"\n{i}/5 Testing {img_name}...")
    
    try:
        result = process_image(img_path)
        
        pred_label = "🔴 MALIGNANT" if result["class"] == 1 else "🟢 BENIGN"
        print(f"  Prediction: {pred_label}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  P(Benign): {result['prob_benign']*100:.1f}%")
        print(f"  P(Malignant): {result['prob_malignant']*100:.1f}%")
        print(f"  Features: {result['features_shape']}")
        print(f"  Explanation: {result['explanation'][:100]}...")
        
        # Save output
        output_name = img_name.replace(".png", ".txt")
        output_path = os.path.join(CONFIG["output_dir"], output_name)
        
        with open(output_path, 'w') as f:
            f.write(f"BreastEdge AI - Test Result\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Image: {img_name}\n")
            f.write(f"Path: {img_path}\n\n")
            f.write(f"PREDICTION\n")
            f.write(f"----------\n")
            f.write(f"Class: {pred_label}\n")
            f.write(f"Confidence: {result['confidence']*100:.1f}%\n\n")
            f.write(f"PROBABILITIES\n")
            f.write(f"-------------\n")
            f.write(f"P(Benign): {result['prob_benign']*100:.1f}%\n")
            f.write(f"P(Malignant): {result['prob_malignant']*100:.1f}%\n\n")
            f.write(f"FEATURES\n")
            f.write(f"--------\n")
            f.write(f"MedSigLIP features: {result['features_shape']}\n\n")
            f.write(f"CLINICAL EXPLANATION (MedGemma 1.5)\n")
            f.write(f"-------------------------------------\n")
            f.write(f"{result['explanation']}\n")
        
        print(f"  ✓ Saved to {output_path}")
        
        results.append({
            "image": img_name,
            "prediction": pred_label,
            "confidence": result["confidence"],
            "success": True
        })
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append({
            "image": img_name,
            "error": str(e),
            "success": False
        })

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

successful = sum(1 for r in results if r.get("success", False))
print(f"\nTotal: {len(demo_images)} images")
print(f"Successful: {successful}")
print(f"Failed: {len(results) - successful}")

print(f"\nOutputs saved to: {CONFIG['output_dir']}")
print("\n✅ BATCH TEST COMPLETE")
