# BreastEdge AI

**Edge-Deployed Breast Cancer Histopathology Classification**

Running 100% locally on NVIDIA DGX Spark — Zero cloud dependency

## Overview

BreastEdge AI is a medical AI system for automated breast cancer detection from histopathology images, powered by three official Google Health AI Developer Foundations (HAI-DEF) models:

- **MedSigLIP-448**: Medical image encoder
- **ResNet50**: Breast cancer classifier (87.92% accuracy)
- **MedGemma 1.5**: Medical explanation AI

## Performance

Trained on 157,572 balanced histopathology images:

- **Accuracy**: 87.92%
- **Sensitivity**: 91.34% (malignant detection)
- **Specificity**: 84.50% (benign detection)
- **AUC-ROC**: 0.9508

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA DGX Spark GB10, 119GB unified memory)
- Ubuntu 22.04+ (or compatible Linux)

### Dependencies

```bash
pip install -r requirements.txt
```

### Models

Models are automatically downloaded from Hugging Face on first run:

- `google/medsiglip-448` (requires Hugging Face account approval)
- `google/medgemma-1.5-4b-it`
- Custom ResNet50 checkpoint: `best_model_full.pth`

## Usage

### Gradio Web Interface

```bash
python3 app.py
```

Access at: `http://localhost:7860`

### Command Line Interface

Test a single image:

```bash
python3 test_cli.py path/to/image.png
```

Test all demo images:

```bash
python3 test_all_demos.py
```

## Project Structure

```
breast_edge_ai/
├── app.py                          # Gradio web interface
├── test_cli.py                     # CLI inference script
├── test_all_demos.py              # Batch test demo images
├── select_demo_images.py          # Select diverse demo images
├── post_production.py             # Video post-production
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── demo_images/                   # 5 diverse test cases
│   ├── demo_benign_1.png
│   ├── demo_benign_2.png
│   ├── demo_malignant_1.png
│   ├── demo_malignant_2.png
│   └── demo_borderline.png
└── demo_outputs/                  # CLI test results
    ├── demo_benign_1.txt
    ├── demo_benign_2.txt
    ├── demo_malignant_1.txt
    ├── demo_malignant_2.txt
    └── demo_borderline.txt
```

## Architecture

### Pipeline Flow

```
Input Image (50x50 PNG)
    ↓
[MedSigLIP-448]
    ↓
Medical Feature Embedding (1152-dim)
    ↓
[ResNet50 Classifier]
    ↓
Binary Classification (Benign/Malignant) + Confidence
    ↓
[MedGemma 1.5]
    ↓
Natural Language Clinical Explanation
```

### Model Details

**MedSigLIP-448**
- Official Google HAI-DEF medical image encoder
- Pre-trained on medical imaging datasets
- Tags: medical, pathology, radiology, dermatology, ophthalmology
- Output: 1152-dimensional embedding

**ResNet50**
- MONAI implementation
- Trained on 157,572 balanced histopathology patches
- 23.5M parameters
- Training splits: 70% train, 15% val, 15% test
- Data augmentation: horizontal + vertical flips
- Optimizer: AdamW (LR 0.0001)
- 5 epochs, batch size 32

**MedGemma 1.5**
- Google's medical language model
- 4B parameters, bfloat16 precision
- Generates clinical explanations for predictions

## Dataset

- **Source**: BreakHis (Breast Cancer Histopathological Database)
- **Total images**: 555,048 PNG patches (50x50 pixels)
- **Classes**:
  - Benign: 397,476 patches
  - Malignant: 157,572 patches
- **Training set**: 157,572 balanced (78,786 each class)

## Training

```bash
python3 full_training_resnet50.py
```

Training configuration:
- 157K balanced images
- ResNet50 architecture
- 5 epochs
- Batch size: 32
- Learning rate: 0.0001
- Optimizer: AdamW
- Data augmentation: random horizontal/vertical flips
- Loss: CrossEntropyLoss

## API Reference

### CLI Interface

```python
python3 test_cli.py <image_path>
```

Output:
- Prediction: BENIGN/MALIGNANT
- Confidence score
- Class probabilities
- MedGemma clinical explanation

### Gradio Interface

Upload histopathology image → Get:
- Visual prediction (Benign/Malignant)
- Confidence bar
- Clinical explanation
- Feature embedding heatmap

## Edge Deployment

**Hardware**: NVIDIA DGX Spark GB10
- 119 GB unified memory
- CUDA 13.0
- Ubuntu 22.04

**Advantages**:
- **100% local inference** — no cloud API calls
- **Zero latency** — no network dependency
- **Data privacy** — patient data never leaves device
- **Offline capable** — works without internet
- **Low cost** — no per-inference API fees

## Limitations

- Trained on BreakHis dataset (specific staining/microscopy setup)
- Binary classification only (benign/malignant)
- No subtype classification
- Requires high-quality 50x50 patches
- **For research and educational purposes only**
- Not FDA-approved for clinical use

## Citation

```bibtex
@software{breastedge_ai_2026,
  author = {Ferhat, Driss Faiz},
  title = {BreastEdge AI: Edge-Deployed Breast Cancer Histopathology Classifier},
  year = {2026},
  publisher = {INKWAY Consulting},
  note = {Google Health AI Developer Foundations Competition Submission}
}
```

## License

MIT License

## Author

**Driss Faiz Ferhat**  
AI Solutions Architect | INKWAY Consulting  
Rabat, Morocco  
[LinkedIn](https://linkedin.com/in/driss-faiz-ferhat) | [GitHub](https://github.com/faizferhat)

## Acknowledgments

- Google Health AI Developer Foundations (HAI-DEF)
- MONAI Framework
- BreakHis Dataset
- Hugging Face Transformers

---

**Competition**: Google Edge AI Prize Track ($100K)  
**Track**: Medical AI on Edge Devices  
**Submission**: February 2026
