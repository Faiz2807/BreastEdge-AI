#!/usr/bin/env python3
"""
BreastEdge AI - Few-Shot Prompting Test
Compare baseline vs few-shot prompts on demo images
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from monai.networks.nets import ResNet
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

# Import prompts module
sys.path.insert(0, os.path.expanduser("~/breast_edge_ai"))
from prompts import get_baseline_prompt, get_fewshot_prompt

CONFIG = {
    "resnet_path": "/home/faiz/monai_breast_classifier_full/best_model_full.pth",
    "medgemma_model": "google/medgemma-1.5-4b-it",
    "demo_images_dir": "/home/faiz/breast_edge_ai/demo_images",
    "output_dir": "/home/faiz/breast_edge_ai/fewshot_results",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224
}

print("=" * 70)
print("BREASTEDGE AI - FEW-SHOT PROMPTING TEST")
print("=" * 70)
print(f"Device: {CONFIG['device']}\n")

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Load models
print("Loading models...")

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

def classify_image(image_path):
    """Classify image with ResNet50."""
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

def get_medgemma_explanation(prompt):
    """Generate MedGemma explanation."""
    inputs = medgemma_tokenizer(prompt, return_tensors="pt").to(medgemma.device)
    
    with torch.no_grad():
        outputs = medgemma.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=medgemma_tokenizer.eos_token_id
        )
    
    response = medgemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response (remove prompt)
    if prompt in response:
        explanation = response.split(prompt)[-1].strip()
    else:
        explanation = response.strip()
    
    return explanation

# Test images
demo_images = [
    ("demo_benign_1.png", "Benign High Confidence"),
    ("demo_benign_2.png", "Benign High Confidence"),
    ("demo_malignant_1.png", "Malignant High Confidence"),
    ("demo_malignant_2.png", "Malignant High Confidence"),
    ("demo_borderline.png", "Borderline Case")
]

print("=" * 70)
print("TESTING PROMPTING STRATEGIES")
print("=" * 70)

all_results = []

for i, (img_name, description) in enumerate(demo_images, 1):
    img_path = os.path.join(CONFIG["demo_images_dir"], img_name)
    
    if not os.path.exists(img_path):
        print(f"\n{i}/5 ⚠️  {img_name} - NOT FOUND")
        continue
    
    print(f"\n{i}/5 Testing {img_name} ({description})...")
    
    # Classify
    prediction = classify_image(img_path)
    pred_label = "MALIGNANT" if prediction["class"] == 1 else "BENIGN"
    
    print(f"  ResNet50: {pred_label} ({prediction['confidence']*100:.1f}% conf)")
    
    # Generate baseline explanation
    print("  Generating BASELINE explanation...", end="", flush=True)
    baseline_prompt = get_baseline_prompt(prediction["class"], prediction["confidence"])
    baseline_explanation = get_medgemma_explanation(baseline_prompt)
    print(" ✓")
    
    # Generate few-shot explanation
    print("  Generating FEW-SHOT explanation...", end="", flush=True)
    fewshot_prompt = get_fewshot_prompt(prediction["class"], prediction["confidence"])
    fewshot_explanation = get_medgemma_explanation(fewshot_prompt)
    print(" ✓")
    
    # Store results
    result = {
        "image": img_name,
        "description": description,
        "prediction": {
            "class": prediction["class"],
            "label": pred_label,
            "confidence": prediction["confidence"],
            "prob_benign": prediction["prob_benign"],
            "prob_malignant": prediction["prob_malignant"]
        },
        "baseline": {
            "prompt": baseline_prompt,
            "explanation": baseline_explanation,
            "length": len(baseline_explanation)
        },
        "fewshot": {
            "prompt": fewshot_prompt,
            "explanation": fewshot_explanation,
            "length": len(fewshot_explanation)
        }
    }
    
    all_results.append(result)
    
    # Save individual result
    output_file = os.path.join(CONFIG["output_dir"], img_name.replace(".png", ".json"))
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ Saved to {output_file}")

# Save comparison summary
print("\n" + "=" * 70)
print("GENERATING COMPARISON REPORT")
print("=" * 70)

report_path = os.path.join(CONFIG["output_dir"], "comparison_report.md")

with open(report_path, 'w') as f:
    f.write("# Few-Shot Prompting Comparison Report\n\n")
    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Models**: ResNet50 + MedGemma 1.5\n\n")
    f.write(f"**Test Images**: {len(all_results)}\n\n")
    f.write("---\n\n")
    
    for result in all_results:
        f.write(f"## {result['image']} ({result['description']})\n\n")
        f.write(f"**Prediction**: {result['prediction']['label']} ")
        f.write(f"({result['prediction']['confidence']*100:.1f}% confidence)\n\n")
        
        f.write("### BASELINE Prompt\n")
        f.write("```\n")
        f.write(result['baseline']['prompt'])
        f.write("\n```\n\n")
        
        f.write("**BASELINE Response**:\n")
        f.write(f"> {result['baseline']['explanation']}\n\n")
        f.write(f"*Length: {result['baseline']['length']} characters*\n\n")
        
        f.write("### FEW-SHOT Prompt\n")
        f.write("```\n")
        f.write(result['fewshot']['prompt'])
        f.write("\n```\n\n")
        
        f.write("**FEW-SHOT Response**:\n")
        f.write(f"> {result['fewshot']['explanation']}\n\n")
        f.write(f"*Length: {result['fewshot']['length']} characters*\n\n")
        
        f.write("### Comparison\n\n")
        baseline_len = result['baseline']['length']
        fewshot_len = result['fewshot']['length']
        f.write(f"- Baseline: {baseline_len} chars\n")
        f.write(f"- Few-shot: {fewshot_len} chars\n")
        f.write(f"- Difference: {fewshot_len - baseline_len:+d} chars\n\n")
        
        f.write("---\n\n")
    
    # Summary statistics
    f.write("## Summary Statistics\n\n")
    
    avg_baseline_len = sum(r['baseline']['length'] for r in all_results) / len(all_results)
    avg_fewshot_len = sum(r['fewshot']['length'] for r in all_results) / len(all_results)
    
    f.write(f"- **Average baseline length**: {avg_baseline_len:.1f} characters\n")
    f.write(f"- **Average few-shot length**: {avg_fewshot_len:.1f} characters\n")
    f.write(f"- **Average difference**: {avg_fewshot_len - avg_baseline_len:+.1f} characters\n\n")
    
    f.write("## Key Observations\n\n")
    f.write("- Few-shot prompts provide structured, example-based context\n")
    f.write("- Baseline prompts are simpler and more direct\n")
    f.write("- Few-shot responses may be more detailed and clinically structured\n")
    f.write("- Baseline responses focus on concise educational explanations\n\n")

print(f"✓ Comparison report saved: {report_path}")

# Save all results to JSON
all_results_path = os.path.join(CONFIG["output_dir"], "all_results.json")
with open(all_results_path, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "MedGemma 1.5",
            "classifier": "ResNet50",
            "test_images": len(all_results)
        },
        "results": all_results
    }, f, indent=2)

print(f"✓ All results saved: {all_results_path}")

print("\n" + "=" * 70)
print("✅ FEW-SHOT PROMPTING TEST COMPLETE")
print("=" * 70)
print(f"\nResults directory: {CONFIG['output_dir']}")
print(f"Files created:")
print(f"  - {len(all_results)} individual JSON results")
print(f"  - comparison_report.md")
print(f"  - all_results.json")
