# BreastEdge AI

**Edge-Deployed Breast Cancer Histopathology Classification with Voice-Powered Clinical Q&A**

Running 100% locally on NVIDIA DGX Spark — Zero cloud dependency — 4 Google HAI-DEF models

## Overview

BreastEdge AI is a medical AI system for automated breast cancer detection from histopathology images, powered by **four Google Health AI Developer Foundations (HAI-DEF) models** in an integrated pipeline:

- **MedSigLIP-448** (0.9B params): Medical image feature extraction
- **ResNet50/MONAI** (23.5M params): Binary classification (benign/malignant)
- **MedGemma 1.5 4B-it** (4.3B params): Clinical explanation generation
- **MedASR** (105M params): Medical speech-to-text for clinician voice interaction

The unique MedASR → MedGemma pipeline enables clinicians to **ask questions by voice** about the analyzed specimen and receive AI-powered medical answers — creating a conversational diagnostic assistant.

## Performance — Independently Verified

Metrics verified through independent testing on 2,000 randomly sampled images (seed 777), separate from training validation:

| Metric | Verified Result | Validation Set |
|--------|----------------|----------------|
| **Accuracy** | **81.85%** | 87.65% |
| **Sensitivity** | **85.30%** | — |
| **Specificity** | **78.40%** | — |
| Test Set Size | 2,000 random | 23,636 (train split) |

> **Transparency note:** Our initial validation set reported 87.65% accuracy. Independent testing on a random global sample confirmed 81.85%. We report the independently verified figures.

## Architecture
```
Input Image (50x50 patch)
    |
    v
MedSigLIP-448 --> Feature extraction (medical visual embeddings)
    |
    v
ResNet50/MONAI --> Classification: BENIGN or MALIGNANT + confidence score
    |
    v
MedGemma 1.5 --> Clinical explanation (3 sections):
                  1. Tissue Architecture
                  2. Cellular Features
                  3. Clinical Correlation
    |
    v
MedASR --> Clinician speaks a question
    |
    v
MedGemma 1.5 --> AI-powered voice Q&A response
```

All four models run simultaneously on a single NVIDIA DGX Spark (128GB unified memory), using ~11.4GB VRAM. Processing time: ~15-25 seconds per image.

## Hardware

| Specification | Value |
|--------------|-------|
| Device | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Grace Blackwell) |
| Memory | 128 GB unified |
| CUDA | 12.8 / Driver 580.95 |
| VRAM used | ~11.4 / 119.7 GB (4 models loaded) |
| Network required | None (100% offline) |
| Power consumption | ~77W |

## Dataset

**Breast Histopathology Images** (Kaggle) — 555,048 patches of 50x50 pixels extracted from 280 patients. Training on 157,572 balanced images (50/50 benign/malignant).

## What We Tried and What Failed

Transparency on our iterative process:

| Approach | Result | Lesson |
|----------|--------|--------|
| MedGemma 4B as direct classifier | 60% sensitivity | VLM not suited for binary classification on small patches |
| LoRA fine-tuning MedGemma | 0% sensitivity (predicted all benign) | 4B model too small for effective fine-tuning on histopathology |
| MedSigLIP + ResNet50 pipeline | **85.3% sensitivity** | Specialized encoder + classifier outperforms generalist VLM |
| MedGemma as explainer (few-shot) | Correct clinical terminology | Best role: interpretation, not classification |
| MedASR + MedGemma voice Q&A | Functional pipeline | Enables natural clinician-AI interaction |

## Why Edge AI Matters for Breast Cancer

- **Morocco has ~150 pathologists** for 37 million people
- Rural clinics have no reliable internet for cloud AI
- Patient data sovereignty requires local processing
- A DGX Spark costs less than one year of cloud GPU rental
- Voice interaction (MedASR) enables hands-free use during microscopy

## Clinical Validation

Clinical validation in progress with Dr. Amal, Chef de Radiologie at CHU Cheikh Khalifa, Casablanca, including testing on 10 clinical histopathology specimens. Results pending.

## Running the Dashboard
```bash
# SSH into DGX Spark
ssh faiz@192.168.1.237

# Terminal 1: Start MedASR micro-service
source ~/medasr_test/bin/activate
python ~/medasr_test/medasr_service.py  # port 7861

# Terminal 2: Start main server
cd ~/breast_edge_ai
python3 server.py  # port 7860

# Open in browser: http://192.168.1.237:7860
```

## Project Structure
```
breast_edge_ai/
├── server.py                    # FastAPI backend (4 HAI-DEF models)
├── static/index.html            # Professional medical dashboard with voice Q&A
├── verify_model.py              # Independent validation script (2000 images)
├── advanced_metrics.py          # ROC, PR, ECE, calibration plots
├── validate_pipeline.py         # End-to-end pipeline validation
├── prompts.py                   # Few-shot prompting templates
├── metrics_plots/               # Publication-ready plots (300 DPI)
└── demo_images/                 # Sample histopathology images

medasr_test/
├── medasr_service.py            # MedASR micro-service (port 7861)
└── (isolated venv)              # Separate environment for MedASR
```

## License

This project was created for the MedGemma Impact Challenge. The models used (MedSigLIP, MedGemma, MedASR) are subject to Google's HAI-DEF licensing terms.

## Author

**Faiz Ferhat** — AI Solutions Architect, Rabat, Morocco

- GitHub: [Faiz2807](https://github.com/Faiz2807)
- Competition: [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
