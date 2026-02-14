#!/usr/bin/env python3
"""
BreastEdge AI - Few-Shot Prompting System for MedGemma 1.5
Enhanced pathology explanations with example-based learning
"""

def get_fewshot_prompt(prediction_class, confidence):
    """
    Generate few-shot prompt for MedGemma 1.5.
    
    Args:
        prediction_class: 0 (benign) or 1 (malignant)
        confidence: float 0-1 (e.g., 0.975)
    
    Returns:
        str: Formatted few-shot prompt
    """
    pred_label = "MALIGNANT (IDC positive)" if prediction_class == 1 else "BENIGN"
    conf_pct = confidence * 100
    
    prompt = f"""You are a board-certified pathologist analyzing breast histopathology patches for invasive ductal carcinoma (IDC).

Example 1: A patch showing uniform, well-organized glandular structures with regular nuclei → Classification: BENIGN

Example 2: A patch showing irregular, densely packed cells with enlarged nuclei and loss of normal architecture → Classification: MALIGNANT (IDC positive)

Now analyze this histopathology patch. The CNN classifier predicted {pred_label} with {conf_pct:.1f}% confidence.

Provide:
1. Your assessment of the tissue patterns visible
2. Whether you agree with the CNN classification
3. Key histological features supporting your assessment

Keep your response concise (2-3 sentences)."""

    return prompt


def get_baseline_prompt(prediction_class, confidence):
    """
    Baseline prompt (current system).
    
    Args:
        prediction_class: 0 (benign) or 1 (malignant)
        confidence: float 0-1
    
    Returns:
        str: Formatted baseline prompt
    """
    pred_label = "malignant" if prediction_class == 1 else "benign"
    conf_pct = confidence * 100
    
    prompt = f"""Medical Education:

Histopathology result: {pred_label.upper()} tissue ({conf_pct:.1f}% confidence)

Provide a 2-sentence clinical explanation of {pred_label} breast tissue features."""

    return prompt


# Prompt templates for different scenarios
PROMPTS = {
    "fewshot": get_fewshot_prompt,
    "baseline": get_baseline_prompt,
}


def get_prompt(style="fewshot", prediction_class=None, confidence=None):
    """
    Get prompt by style.
    
    Args:
        style: "fewshot" or "baseline"
        prediction_class: 0 or 1
        confidence: float 0-1
    
    Returns:
        str: Formatted prompt
    """
    if style not in PROMPTS:
        raise ValueError(f"Unknown prompt style: {style}")
    
    return PROMPTS[style](prediction_class, confidence)


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("BREASTEDGE AI - PROMPTING SYSTEM")
    print("=" * 70)
    
    print("\nBASELINE PROMPT (Benign, 97.5%):")
    print("-" * 70)
    print(get_baseline_prompt(0, 0.975))
    
    print("\n\nFEW-SHOT PROMPT (Benign, 97.5%):")
    print("-" * 70)
    print(get_fewshot_prompt(0, 0.975))
    
    print("\n\nBASELINE PROMPT (Malignant, 99.9%):")
    print("-" * 70)
    print(get_baseline_prompt(1, 0.999))
    
    print("\n\nFEW-SHOT PROMPT (Malignant, 99.9%):")
    print("-" * 70)
    print(get_fewshot_prompt(1, 0.999))
