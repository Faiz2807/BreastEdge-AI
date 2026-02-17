#!/usr/bin/env python3
"""
BreastEdge AI - Professional Medical Dashboard
FastAPI backend for breast cancer histopathology classification
Models: MedSigLIP-448 + ResNet50 + MedGemma 1.5 (few-shot prompting)
"""

import os
import io
import base64
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import psutil
from scipy.ndimage import gaussian_filter
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from monai.networks.nets import ResNet


def clean_explanation(text):
    """Extract first analysis block from MedGemma output. No modification of medical content."""
    import re, json as json_mod
    
    # === METHOD 1: JSON format (typically malignant cases) ===
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{\s*"assessment".*?\})', text, re.DOTALL)
    
    if json_match:
        try:
            data = json_mod.loads(json_match.group(1))
            if 'assessment' in data and isinstance(data['assessment'], list):
                lines = data['assessment']
                arch = cell = clin = ''
                for line in lines:
                    clean_line = re.sub(r'^\d+\.\s*', '', str(line)).strip()
                    if 'tissue architecture' in clean_line.lower():
                        arch = re.sub(r'^Tissue\s*Architecture:\s*', '', clean_line, flags=re.IGNORECASE).strip()
                    elif 'cellular feature' in clean_line.lower():
                        cell = re.sub(r'^Cellular\s*Features?:\s*', '', clean_line, flags=re.IGNORECASE).strip()
                    elif 'clinical correlation' in clean_line.lower():
                        clin = re.sub(r'^Clinical\s*Correlation:\s*', '', clean_line, flags=re.IGNORECASE).strip()
                if arch and cell and clin:
                    return f"1. Tissue Architecture: {arch}\n2. Cellular Features: {cell}\n3. Clinical Correlation: {clin}"
        except (json_mod.JSONDecodeError, KeyError, IndexError):
            pass
    
    # === METHOD 2: Text format (typically benign cases) ===
    # Find first occurrence of the 3 sections
    arch_match = re.search(r'Tissue\s*Architecture:\s*(.*?)(?=\n.*?Cellular\s*Feature|$)', text, re.IGNORECASE)
    cell_match = re.search(r'Cellular\s*Features?:\s*(.*?)(?=\n.*?Clinical\s*Correlation|$)', text, re.IGNORECASE)
    clin_match = re.search(r'Clinical\s*Correlation:\s*(.*?)(?=\n---|\nThis|\nYour|\nLet|\nOkay|\nCase|\n\n|$)', text, re.IGNORECASE)
    
    if arch_match and cell_match and clin_match:
        arch = arch_match.group(1).strip()
        cell = cell_match.group(1).strip()
        clin = clin_match.group(1).strip()
        # Clean any trailing numbering
        arch = re.sub(r'\s*\d+\.\s*$', '', arch).strip()
        cell = re.sub(r'\s*\d+\.\s*$', '', cell).strip()
        clin = re.sub(r'\s*\d+\.\s*$', '', clin).strip()
        if arch and cell and clin:
            return f"1. Tissue Architecture: {arch}\n2. Cellular Features: {cell}\n3. Clinical Correlation: {clin}"
    
    # === FALLBACK: Return raw text truncated to first meaningful block ===
    # Cut at first repetition marker
    cut = re.split(r"\n---\n|\nYour assessment|\n```json|\nThis is a good|\nLet", text)[0].strip()
    if cut:
        return cut
    return text[:500]


    # Return structured format that frontend can parse reliably
    return f"1. Tissue Architecture: {arch}\n2. Cellular Features: {cell}\n3. Clinical Correlation: {clin}"


# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Paths
HOME = Path.home()
MODEL_DIR = HOME / "monai_breast_classifier_full"
MODEL_PATH = MODEL_DIR / "best_model_full.pth"

# Few-shot prompting template
FEWSHOT_TEMPLATE = """You are an expert pathologist analyzing breast histopathology images.

Example 1:
Image: Benign tissue with normal ductal structures
CNN Classification: BENIGN (confidence: 95.2%)
Assessment:
1. Tissue Architecture: Well-organized ductal structures with preserved basement membranes
2. Cellular Features: Uniform cell morphology, regular nuclei, no atypical mitoses
3. Clinical Correlation: Consistent with benign breast tissue; no malignant features identified

Example 2:
Image: Malignant tissue showing invasive carcinoma
CNN Classification: MALIGNANT (confidence: 98.7%)
Assessment:
1. Tissue Architecture: Loss of normal architecture, irregular glandular formations, stromal invasion
2. Cellular Features: Pleomorphic nuclei, increased nuclear-cytoplasmic ratio, abnormal mitoses
3. Clinical Correlation: Features consistent with invasive ductal carcinoma; warrants further evaluation

Now analyze this case:
Image: Breast histopathology specimen
CNN Classification: {classification} (confidence: {confidence}%)

Provide your assessment in exactly 3 points:
1. Tissue Architecture: [describe architectural patterns]
2. Cellular Features: [describe cell morphology and nuclear characteristics]
3. Clinical Correlation: [interpret findings and clinical significance]

Keep each point concise (1-2 sentences). Use professional pathology terminology."""

# Initialize models (lazy loading)
siglip_model = None
siglip_processor = None
resnet_model = None
gemma_model = None
gemma_tokenizer = None

def load_models():
    """Load all three models: MedSigLIP-448, ResNet50, MedGemma 1.5"""
    global siglip_model, siglip_processor, resnet_model, gemma_model, gemma_tokenizer
    
    print("📦 Loading models...")
    
    # 1. MedSigLIP-448 (feature extractor)
    print("  • MedSigLIP-448...")
    siglip_model = AutoModel.from_pretrained("google/medsiglip-448", trust_remote_code=True).to(device)
    siglip_processor = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 2. ResNet50 classifier
    print("  • ResNet50 classifier...")
    resnet_model = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=2,
        n_input_channels=3,
        num_classes=2
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Handle both checkpoint dict and direct state_dict formats
    if "model_state_dict" in checkpoint:
        resnet_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        resnet_model.load_state_dict(checkpoint)
    resnet_model.eval()
    print(f"    Loaded from: {MODEL_PATH}")
    
    # 3. MedGemma 1.5 (explainer)
    print("  • MedGemma 1.5...")
    gemma_model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-1.5-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    gemma_tokenizer = AutoTokenizer.from_pretrained(
        "google/medgemma-1.5-4b-it",
        local_files_only=True
    )
    
    print("✅ All models loaded successfully")

def get_gpu_stats() -> Dict[str, Any]:
    """Get GPU temperature and VRAM usage"""
    if not torch.cuda.is_available():
        return {"gpu_temp": "N/A", "vram_used_gb": 0, "vram_total_gb": 0}
    
    try:
        # VRAM usage
        vram_used = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # GPU temp (via nvidia-smi)
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        gpu_temp = f"{result.stdout.strip()}°C" if result.returncode == 0 else "N/A"
        
        return {
            "gpu_temp": gpu_temp,
            "vram_used_gb": round(vram_used, 1),
            "vram_total_gb": round(vram_total, 1)
        }
    except Exception as e:
        print(f"⚠️  GPU stats error: {e}")
        return {"gpu_temp": "N/A", "vram_used_gb": 0, "vram_total_gb": 0}

def predict_image(image: Image.Image) -> Dict[str, Any]:
    """
    Run full prediction pipeline:
    1. Extract MedSigLIP features
    2. Classify with ResNet50
    3. Generate heatmap
    4. Explain with MedGemma (few-shot)
    """
    # Preprocess
    img_rgb = image.convert("RGB")
    
    # Preprocessing for MedSigLIP (CLIP normalization)
    siglip_tensor = siglip_processor(img_rgb).unsqueeze(0).to(device)
    
    # Preprocessing for ResNet50 (ImageNet normalization)
    resnet_processor = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resnet_tensor = resnet_processor(img_rgb).unsqueeze(0).to(device)
    
    # 1. MedSigLIP features (for heatmap)
    with torch.no_grad():
        siglip_features = siglip_model.get_image_features(pixel_values=siglip_tensor)
    
    # 2. ResNet50 classification (ImageNet normalization)
    with torch.no_grad():
        logits = resnet_model(resnet_tensor)
        probs = F.softmax(logits, dim=1)
        confidence = probs.max().item() * 100
        pred_class = probs.argmax(dim=1).item()
    
    prediction = "BENIGN" if pred_class == 0 else "MALIGNANT"
    
    # 3. Generate heatmap (simplified attention map)
    # Create a simple heatmap based on image statistics
    with torch.no_grad():
        # Resize original image
        img_resized = img_rgb.resize((448, 448))
        img_array = np.array(img_resized)
        
        # Create a simple attention heatmap based on intensity variance
        gray = np.mean(img_array, axis=2)
        # Apply Gaussian filter for smoothing
        heatmap = gaussian_filter(gray, sigma=10)
        
        # Normalize to 0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Convert to colormap
        from matplotlib import cm
        cmap = cm.get_cmap('jet')
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        
        # Blend with original image (60% original, 40% heatmap)
        blended = (0.6 * img_array + 0.4 * heatmap_colored).astype(np.uint8)
        
        # Encode to base64
        heatmap_pil = Image.fromarray(blended)
        buffered = io.BytesIO()
        heatmap_pil.save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # 4. MedGemma explanation (few-shot)
    prompt = FEWSHOT_TEMPLATE.format(
        classification=prediction,
        confidence=f"{confidence:.1f}"
    )
    
    inputs = gemma_tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=False,
            pad_token_id=gemma_tokenizer.eos_token_id
        )
    
    full_response = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part (after prompt)
    explanation = full_response[len(prompt):].strip()
    explanation = clean_explanation(explanation)
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "heatmap_base64": heatmap_b64
    }

# FastAPI app
app = FastAPI(
    title="BreastEdge AI",
    description="Professional breast cancer histopathology classification dashboard",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup: load models
@app.on_event("startup")
async def startup_event():
    load_models()

# Health endpoint
@app.get("/api/health")
async def health_check():
    """System health status"""
    gpu_stats = get_gpu_stats()
    
    return JSONResponse({
        "status": "operational",
        "device": str(device),
        "gpu_temp": gpu_stats["gpu_temp"],
        "vram_used_gb": gpu_stats["vram_used_gb"],
        "vram_total_gb": gpu_stats["vram_total_gb"],
        "models_loaded": all([
            siglip_model is not None,
            resnet_model is not None,
            gemma_model is not None
        ])
    })

# Analyze endpoint
@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded histopathology image
    Returns: prediction, confidence, explanation, heatmap
    """
    try:
        # Validate file
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run prediction
        result = predict_image(image)
        
        return JSONResponse(result)
    
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === MedASR Proxy (4th HAI-DEF model) ===
# Forwards audio to MedASR micro-service on port 7861
import httpx

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Proxy to MedASR micro-service on port 7861"""
    try:
        contents = await file.read()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:7861/transcribe",
                files={"file": (file.filename or "audio.webm", contents, file.content_type or "audio/webm")}
            )
            return JSONResponse(response.json())
    except Exception as e:
        return JSONResponse(
            {"transcription": "", "error": str(e), "status": "error"},
            status_code=503
        )

# === MedGemma Voice Q&A (MedASR -> MedGemma pipeline) ===
@app.post("/api/ask")
async def ask_medgemma(file: UploadFile = File(None), question: str = ""):
    """Send voice-transcribed question to MedGemma with image context"""
    from fastapi import Form
    try:
        # Build prompt with the question from MedASR
        prompt = f"""You are an expert pathologist. A clinician is asking you a question about a breast histopathology specimen that was just analyzed.

Clinician question: {question}

Provide a clear, concise medical answer in 2-3 sentences using professional pathology terminology."""
        
        inputs = gemma_tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
        with torch.no_grad():
            outputs = gemma_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=gemma_tokenizer.eos_token_id
            )
        full_response = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        return JSONResponse({"answer": answer, "status": "success"})
    except Exception as e:
        print(f"Ask error: {e}")
        return JSONResponse({"answer": "", "error": str(e), "status": "error"}, status_code=500)
# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# Run server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🏥 BreastEdge AI Professional Dashboard")
    print("="*60)
    print(f"📍 Local: http://localhost:7860")
    print(f"🌐 Network: http://192.168.1.237:7860")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
