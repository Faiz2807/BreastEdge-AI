# BreastEdge AI

**Edge-Deployed Breast Cancer Histopathology Classification**

Running 100% locally on NVIDIA DGX Spark — Zero cloud dependency

## Overview

BreastEdge AI is a medical AI system for automated breast cancer detection from histopathology images, powered by three Google Health AI Developer Foundations (HAI-DEF) models in a sequential pipeline:

- **MedSigLIP-448** (0.9B params): Medical image feature extraction
- **ResNet50/MONAI** (23.5M params): Binary classification (benign/malignant)
- **MedGemma 1.5 4B-it** (4.3B params): Clinical explanation generation

## Performance — Independently Verified

Metrics verified through independent testing on 2,000 randomly sampled images (seed 777), separate from training validation:

| Metric | Verified Result | Validation Set |
|--------|----------------|----------------|
| **Accuracy** | **81.85%** | 87.65% |
| **Sensitivity** | **85.30%** | — |
| **Specificity** | **78.40%** | — |
| Test Set Size | 2,000 random | 23,636 (train split) |

> **Transparency note:** Our initial validation set reported 87.65% accuracy. Independent testing on a random global sample confirmed 81.85%. We report the independently verified figures. The ~6-point gap is consistent with expected generalization from validation to held-out test data.

## Architecture

```
Input Image (50×50 patch)
    ↓
MedSigLIP-448 → Feature extraction (medical visual embeddings)
    ↓
ResNet50/MONAI → Classification: BENIGN or MALIGNANT + confidence score
    ↓
MedGemma 1.5 → Clinical explanation in 3 sections:
                 1. Tissue Architecture
                 2. Cellular Features
                 3. Clinical Correlation
```

All three models run simultaneously on a single NVIDIA DGX Spark (128GB unified memory), using only 11.4GB VRAM. Processing time: ~15-25 seconds per image.

## Hardware

| Specification | Value |
|--------------|-------|
| Device | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Grace Blackwell) |
| Memory | 128 GB unified |
| CUDA | 12.8 / Driver 580.95 |
| VRAM used | 11.4 / 119.7 GB (3 models loaded) |
| Network required | None (100% offline) |

## Dataset

**Breast Histopathology Images** (Kaggle) — 555,048 patches of 50×50 pixels extracted from 280 patients. Training on 157,572 balanced images (50/50 benign/malignant).

## What We Tried and What Failed

Transparency on our iterative process:

| Approach | Result | Lesson |
|----------|--------|--------|
| MedGemma 4B as direct classifier | 60% sensitivity | VLM not suited for binary classification on small patches |
| LoRA fine-tuning MedGemma | 0% sensitivity (predicted all benign) | 4B model too small for effective fine-tuning on histopathology |
| MedSigLIP + ResNet50 pipeline | **85.3% sensitivity** | Specialized encoder + classifier outperforms generalist VLM |
| MedGemma as explainer (few-shot) | Correct clinical terminology | Best role: interpretation, not classification |

## Why Edge AI Matters for Breast Cancer

- **Morocco has ~150 pathologists** for 37 million people
- Rural clinics have no reliable internet for cloud AI
- Patient data sovereignty requires local processing
- A DGX Spark costs less than one year of cloud GPU rental

## Clinical Validation

Planned validation with Dr. Amal, Head of Radiology at CHU Cheikh Khalifa (Rabat), including testing on 10 clinical histopathology specimens. Results pending.

## Running the Dashboard

```bash
# SSH into DGX Spark
ssh faiz@192.168.1.237

# Start the server
cd ~/breast_edge_ai
python3 server.py

# Open in browser
# http://192.168.1.237:7860
```

## Project Structure

```
breast_edge_ai/
├── server.py                    # FastAPI backend (3 HAI-DEF models)
├── static/index.html            # Professional medical dashboard
├── verify_model.py              # Independent validation script (2000 images)
├── advanced_metrics.py          # ROC, PR, ECE, calibration plots
├── validate_pipeline.py         # End-to-end pipeline validation
├── metrics_plots/               # Publication-ready plots (300 DPI)
└── demo_images/                 # Sample histopathology images
```

## License

This project was created for the MedGemma Impact Challenge. The models used (MedSigLIP, MedGemma) are subject to Google's HAI-DEF licensing terms.

## Author

**Faiz Ferhat** — AI Solutions Architect, Rabat, Morocco

- GitHub: [Faiz2807](https://github.com/Faiz2807)
- Competition: [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
