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

## Advanced Metrics

Comprehensive evaluation on full test set (41,627 images):

### Model Discrimination
- **ROC AUC**: 0.9485 — Excellent discrimination ability
- **Average Precision**: 0.8702 — Strong performance on imbalanced dataset

### Test Performance
- **Accuracy**: 86.06%
- **Sensitivity**: 93.68% (malignant detection)
- **Specificity**: 83.04% (benign detection)

### Model Calibration
- **Expected Calibration Error (ECE)**: 0.1315
  - Indicates moderate over-confidence (threshold: <0.05 well-calibrated)
  - Recommendation: Apply temperature scaling for clinical deployment
  - Impact: Confidence scores require calibration before clinical use

### Visualization Plots

Publication-ready plots (300 DPI) available in `metrics_plots/`:

1. **ROC Curve** (`roc_curve.png`)
   - Shows true positive rate vs false positive rate
   - AUC 0.9485 indicates excellent model discrimination

2. **Precision-Recall Curve** (`precision_recall_curve.png`)
   - Critical for imbalanced medical datasets
   - Average Precision 0.8702

3. **Confidence Distribution** (`confidence_distribution.png`)
   - Separate histograms for benign vs malignant predictions
   - Shows model confidence separation between classes

4. **Calibration Plot** (`calibration_plot.png`)
   - Compares predicted confidence to actual accuracy
   - ECE 0.1315 visible as deviation from diagonal
   - Suggests need for post-hoc calibration

### Clinical Interpretation

- **Excellent discrimination** (AUC > 0.94): Model reliably separates benign from malignant
- **High sensitivity** (93.68%): Catches most cancer cases (low false negatives)
- **Good specificity** (83.04%): Acceptable false positive rate for screening
- **Calibration needed**: Confidence scores not yet reliable for clinical decision-making

For detailed metrics, see `metrics_plots/advanced_metrics.json`.



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

## Few-shot Prompting

BreastEdge AI uses **few-shot prompting** to improve the quality and consistency of MedGemma 1.5 explanations.

### Approach

Instead of simple zero-shot prompts, we provide MedGemma with:
1. **Role establishment**: "You are a board-certified pathologist"
2. **Clinical examples**: 2 concrete cases (benign vs malignant features)
3. **Structured output request**: 3-point assessment format
4. **CNN prediction context**: Includes classifier confidence

### Prompt Template

```
You are a board-certified pathologist analyzing breast histopathology patches 
for invasive ductal carcinoma (IDC).

Example 1: A patch showing uniform, well-organized glandular structures with 
regular nuclei → Classification: BENIGN

Example 2: A patch showing irregular, densely packed cells with enlarged nuclei 
and loss of normal architecture → Classification: MALIGNANT (IDC positive)

Now analyze this histopathology patch. The CNN classifier predicted [PREDICTION] 
with [CONFIDENCE]% confidence.

Provide:
1. Your assessment of the tissue patterns visible
2. Whether you agree with the CNN classification  
3. Key histological features supporting your assessment

Keep your response concise (2-3 sentences).
```

### Benefits Over Baseline

Comparison on 5 test images (see `fewshot_results/`):

| Metric | Baseline | Few-shot | Improvement |
|--------|----------|----------|-------------|
| **Avg. length** | 842 chars | 497 chars | **41% more concise** |
| **Agreement accuracy** | 80% (1 error) | 100% | **More reliable** |
| **Structure** | Variable | Consistent 3-point | **Better format** |
| **Terminology** | Educational | Clinical pathology | **More professional** |

### Key Improvements

1. **Reliability**: 100% agreement with CNN predictions vs 80% baseline
2. **Conciseness**: 41% shorter responses while maintaining clinical value
3. **Structure**: Consistent 3-point format every time
4. **Terminology**: Uses professional pathology language ("nuclear pleomorphism", "mitotic activity")
5. **Borderline cases**: Better handling of uncertain predictions (69.6% confidence)

### Implementation

Few-shot prompting is implemented in `prompts.py` and integrated into the Gradio interface (`app.py`):

```python
from prompts import get_fewshot_prompt

# Generate structured clinical explanation
prompt = get_fewshot_prompt(prediction_class, confidence)
explanation = medgemma.generate(prompt)
```

For detailed comparison and results, see:
- `fewshot_results/comparison_report.md` — Side-by-side comparison
- `fewshot_results/FEWSHOT_RESULTS_SUMMARY.md` — Analysis and recommendations
- `test_fewshot.py` — Reproducible testing script





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
