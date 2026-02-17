#!/usr/bin/env python3
"""
BreastEdge AI - Medical Pathology Classification Demo
Running 100% locally on NVIDIA DGX Spark - Zero cloud dependency

Author: Faiz Ferhat, INKWAY Consulting
Date: 2026-02-14

Google HAI-DEF Models:
- MedSigLIP-448 (Medical Image Encoder)
- ResNet50 (Breast Cancer Classifier)
- MedGemma 1.5 (Medical Explanation AI)
"""

import gradio as gr
from prompts import get_fewshot_prompt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import io

from monai.networks.nets import ResNet
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "resnet_path": "/home/faiz/monai_breast_classifier_full/best_model_full.pth",
    "medgemma_model": "google/medgemma-1.5-4b-it",
    "medsiglip_model": "google/medsiglip-448",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 224
}

print("=" * 70)
print("BREASTEDGE AI - INITIALIZING")
print("=" * 70)
print(f"Device: {CONFIG['device']}")

# ============================================================================
# LOAD MODELS
# ============================================================================

print("\n📥 Loading MedSigLIP-448...")
medsiglip_processor = AutoProcessor.from_pretrained(CONFIG["medsiglip_model"])
medsiglip_model = AutoModel.from_pretrained(
    CONFIG["medsiglip_model"],
    torch_dtype=torch.float16
).to(CONFIG["device"])
medsiglip_model.eval()
print("✓ MedSigLIP-448 loaded")

print("\n📥 Loading ResNet50...")
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

print("\n📥 Loading MedGemma 1.5...")
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
print("✓ MedGemma 1.5 loaded")

# ============================================================================
# TRANSFORMS
# ============================================================================

resnet_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def extract_medsiglip_features(image):
    """Extract MedSigLIP features."""
    inputs = medsiglip_processor(images=image, return_tensors="pt").to(CONFIG["device"])
    with torch.no_grad():
        vision_outputs = medsiglip_model.vision_model(**inputs)
        features = vision_outputs.pooler_output
    return features.cpu().numpy()

def classify_image(image):
    """Classify with ResNet50."""
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

def generate_explanation(prediction):
    """Generate MedGemma explanation using few-shot prompting."""
    # Use few-shot prompt from prompts.py
    prompt = get_fewshot_prompt(prediction["class"], prediction["confidence"])
    
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
    
    # Extract only response (remove prompt)
    if prompt in response:
        explanation = response.split(prompt)[-1].strip()
    else:
        explanation = response.strip()
    
    # Truncate to reasonable length
    if len(explanation) > 500:
        explanation = explanation[:500] + "..."
    
    return explanation

def visualize_features(features):
    """Create heatmap visualization of MedSigLIP features."""
    features_2d = features.reshape(16, -1)[:16, :32]  # Truncate for visualization
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(features_2d, cmap='viridis', aspect='auto')
    ax.set_title('MedSigLIP-448 Medical Feature Embedding', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Feature Layer', fontsize=12)
    plt.colorbar(im, ax=ax, label='Activation')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def predict(image):
    """Main prediction pipeline."""
    if image is None:
        return "Please upload an image", "", None, None
    
    # Convert to PIL
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    
    # Extract features
    features = extract_medsiglip_features(image)
    
    # Classify
    prediction = classify_image(image)
    
    # Generate explanation
    explanation = generate_explanation(prediction)
    
    # Create visualizations
    feature_viz = visualize_features(features)
    
    # Format prediction
    pred_label = "🔴 MALIGNANT" if prediction["class"] == 1 else "🟢 BENIGN"
    confidence = prediction["confidence"] * 100
    
    prediction_text = f"""
## {pred_label}

**Confidence:** {confidence:.1f}%

**Probabilities:**
- Benign: {prediction['prob_benign']*100:.1f}%
- Malignant: {prediction['prob_malignant']*100:.1f}%
"""
    
    # Confidence bar
    confidence_bar = f"""
<div style="background-color: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 10px 0;">
    <div style="background-color: {'#ff4444' if prediction['class'] == 1 else '#44ff44'}; height: 30px; width: {confidence:.1f}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
        {confidence:.1f}%
    </div>
</div>
"""
    
    return prediction_text, explanation, feature_viz, confidence_bar

# ============================================================================
# GRADIO APP
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="BreastEdge AI") as demo:
    gr.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 2.5em; font-weight: bold;">🔬 BreastEdge AI</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">
                Medical Breast Histopathology Classification
            </p>
            <p style="color: #e0e0e0; margin: 10px 0 0 0; font-size: 1em;">
                ⚡ Running 100% locally on NVIDIA DGX Spark — Zero cloud dependency
            </p>
            <p style="color: #d0d0d0; margin: 5px 0 0 0; font-size: 0.9em;">
                Powered by Google HAI-DEF: MedSigLIP-448 • ResNet50 • MedGemma 1.5
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Histopathology Image",
                type="pil",
                height=400
            )
            predict_btn = gr.Button("🔍 Analyze Tissue", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📋 Instructions
            1. Upload a breast histopathology image (PNG, JPG)
            2. Click **Analyze Tissue** to run AI classification
            3. View prediction, confidence, explanation & features
            
            ### 🎯 Model Info
            - **Classifier:** ResNet50 (81.85% accuracy)
            - **Training:** 157,572 balanced images
            - **Metrics:** 85.30% sensitivity, 78.40% specificity
            """)
        
        with gr.Column(scale=2):
            prediction_output = gr.Markdown(label="Prediction")
            confidence_bar = gr.HTML(label="Confidence")
            explanation_output = gr.Textbox(
                label="🤖 MedGemma 1.5 Clinical Explanation",
                lines=5,
                max_lines=10
            )
            feature_viz = gr.Image(label="🧠 MedSigLIP-448 Feature Embedding")
    
    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[prediction_output, explanation_output, feature_viz, confidence_bar]
    )
    
    gr.Markdown("""
    ---
    **BreastEdge AI** | INKWAY Consulting | Faiz Ferhat  
    Medical AI powered by Google Health AI Developer Foundations  
    *For research and educational purposes only. Not for clinical use.*
    """)

print("\n" + "=" * 70)
print("✅ BreastEdge AI Ready")
print("=" * 70)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
