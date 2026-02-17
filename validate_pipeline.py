import requests
import os
import random
import json
import time

API_URL = "http://localhost:7860/api/analyze"
BASE_DIR = os.path.expanduser("~/breast_data")

# Collect all images by class
benign_images = []
malignant_images = []

for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        if f.endswith('.png'):
            path = os.path.join(root, f)
            if '/0/' in path:
                benign_images.append(path)
            elif '/1/' in path:
                malignant_images.append(path)

print(f"Found {len(benign_images)} benign, {len(malignant_images)} malignant")

# Sample 50 of each
random.seed(42)
test_benign = random.sample(benign_images, min(50, len(benign_images)))
test_malignant = random.sample(malignant_images, min(50, len(malignant_images)))

results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "errors": 0, "times": []}

def test_image(path, expected):
    global results
    try:
        start = time.time()
        with open(path, 'rb') as f:
            resp = requests.post(API_URL, files={"file": f}, timeout=120)
        elapsed = time.time() - start
        results["times"].append(elapsed)
        
        data = resp.json()
        pred = data.get("prediction", "UNKNOWN")
        conf = data.get("confidence", 0)
        
        if expected == "MALIGNANT":
            if pred == "MALIGNANT":
                results["TP"] += 1
            else:
                results["FN"] += 1
        else:
            if pred == "BENIGN":
                results["TN"] += 1
            else:
                results["FP"] += 1
        
        return pred, conf, elapsed
    except Exception as e:
        results["errors"] += 1
        return "ERROR", 0, 0

print("\n=== Testing 50 MALIGNANT images ===")
for i, path in enumerate(test_malignant):
    pred, conf, t = test_image(path, "MALIGNANT")
    status = "OK" if pred == "MALIGNANT" else "MISS"
    print(f"  [{i+1}/50] {status} - {pred} {conf:.1f}% ({t:.1f}s)")

print("\n=== Testing 50 BENIGN images ===")
for i, path in enumerate(test_benign):
    pred, conf, t = test_image(path, "BENIGN")
    status = "OK" if pred == "BENIGN" else "MISS"
    print(f"  [{i+1}/50] {status} - {pred} {conf:.1f}% ({t:.1f}s)")

# Calculate metrics
total = results["TP"] + results["TN"] + results["FP"] + results["FN"]
accuracy = (results["TP"] + results["TN"]) / total * 100 if total else 0
sensitivity = results["TP"] / (results["TP"] + results["FN"]) * 100 if (results["TP"] + results["FN"]) else 0
specificity = results["TN"] / (results["TN"] + results["FP"]) * 100 if (results["TN"] + results["FP"]) else 0
avg_time = sum(results["times"]) / len(results["times"]) if results["times"] else 0

print("\n" + "="*50)
print("PIPELINE VALIDATION RESULTS (100 random images)")
print("="*50)
print(f"TP: {results['TP']}  FN: {results['FN']}  (malignant)")
print(f"TN: {results['TN']}  FP: {results['FP']}  (benign)")
print(f"Errors: {results['errors']}")
print(f"Accuracy:    {accuracy:.1f}%")
print(f"Sensitivity: {sensitivity:.1f}%")
print(f"Specificity: {specificity:.1f}%")
print(f"Avg time:    {avg_time:.1f}s per image")
print(f"Total time:  {sum(results['times']):.0f}s")

# Save results
with open("validation_results.json", "w") as f:
    json.dump({
        "confusion_matrix": results,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "avg_inference_time": avg_time,
        "total_images": total
    }, f, indent=2)

print("\n✅ Results saved to validation_results.json")
