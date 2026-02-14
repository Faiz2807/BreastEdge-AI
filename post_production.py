#!/usr/bin/env python3
"""
BreastEdge AI - Video Post-Production
Automatically adds title, outro, and concatenates demo video
"""

import os
import subprocess
import json

BASE_DIR = "/home/faiz/breast_edge_ai"
RAW_VIDEO = f"{BASE_DIR}/demo_raw.mp4"
TITLE_VIDEO = f"{BASE_DIR}/title.mp4"
OUTRO_VIDEO = f"{BASE_DIR}/outro.mp4"
FINAL_VIDEO = f"{BASE_DIR}/demo_final.mp4"
CONCAT_FILE = f"{BASE_DIR}/concat.txt"

print("=" * 70)
print("BREASTEDGE AI - VIDEO POST-PRODUCTION")
print("=" * 70)

# Check if raw video exists
if not os.path.exists(RAW_VIDEO):
    print(f"\n❌ Error: {RAW_VIDEO} not found!")
    print("Please record the demo first.")
    exit(1)

# Check raw video duration
print("\n📊 Checking raw video duration...")
result = subprocess.run([
    "ffprobe", "-v", "error", "-show_entries",
    "format=duration", "-of", "json", RAW_VIDEO
], capture_output=True, text=True)

duration_data = json.loads(result.stdout)
raw_duration = float(duration_data["format"]["duration"])
print(f"Raw video duration: {raw_duration:.1f} seconds ({raw_duration/60:.1f} minutes)")

if raw_duration > 170:  # Leave room for title + outro (10 sec)
    print(f"⚠️  Warning: Raw video is {raw_duration:.1f}s (>{170}s). May exceed 3 min limit.")

# Create title screen (5 seconds)
print("\n🎬 Creating title screen...")
title_cmd = [
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", "color=c=0x1A73E8:s=1920x1080:d=5",
    "-vf",
    "drawtext=text='BreastEdge AI':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-50,"
    "drawtext=text='Edge-Deployed Breast Cancer Histopathology Classifier':fontsize=32:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+30,"
    "drawtext=text='MedSigLIP + ResNet50 + MedGemma 1.5 | NVIDIA DGX Spark':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+80,"
    "drawtext=text='Driss Faiz Ferhat - AI Solutions Architect':fontsize=20:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+130",
    "-c:v", "libx264", "-t", "5", TITLE_VIDEO
]
subprocess.run(title_cmd, check=True)
print(f"✓ Title created: {TITLE_VIDEO}")

# Create outro screen (5 seconds)
print("\n🎬 Creating outro screen...")
outro_cmd = [
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", "color=c=0x1A73E8:s=1920x1080:d=5",
    "-vf",
    "drawtext=text='Results: 87.92%% Accuracy | 91.34%% Sensitivity | AUC 0.9508':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-60,"
    "drawtext=text='3 Google HAI-DEF Models | 100%% Local | Zero Cloud':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+10,"
    "drawtext=text='Edge AI Prize Track | INKWAY Consulting':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2+60",
    "-c:v", "libx264", "-t", "5", OUTRO_VIDEO
]
subprocess.run(outro_cmd, check=True)
print(f"✓ Outro created: {OUTRO_VIDEO}")

# Create concatenation file
print("\n🔗 Concatenating videos...")
with open(CONCAT_FILE, 'w') as f:
    f.write(f"file '{TITLE_VIDEO}'\n")
    f.write(f"file '{RAW_VIDEO}'\n")
    f.write(f"file '{OUTRO_VIDEO}'\n")

# Concatenate
concat_cmd = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", CONCAT_FILE, "-c", "copy", FINAL_VIDEO
]
subprocess.run(concat_cmd, check=True)
print(f"✓ Videos concatenated: {FINAL_VIDEO}")

# Check final duration
print("\n📊 Checking final video duration...")
result = subprocess.run([
    "ffprobe", "-v", "error", "-show_entries",
    "format=duration,size", "-of", "json", FINAL_VIDEO
], capture_output=True, text=True)

final_data = json.loads(result.stdout)
final_duration = float(final_data["format"]["duration"])
final_size_mb = int(final_data["format"]["size"]) / (1024 * 1024)

print(f"Final duration: {final_duration:.1f} seconds ({final_duration/60:.2f} minutes)")
print(f"Final size: {final_size_mb:.1f} MB")

if final_duration > 180:
    print(f"\n⚠️  Video exceeds 3 minutes! Trimming to 180 seconds...")
    trimmed_video = f"{BASE_DIR}/demo_3min.mp4"
    trim_cmd = [
        "ffmpeg", "-y", "-i", FINAL_VIDEO,
        "-t", "180", "-c", "copy", trimmed_video
    ]
    subprocess.run(trim_cmd, check=True)
    print(f"✓ Trimmed video: {trimmed_video}")
    print(f"\n✅ FINAL VIDEO: {trimmed_video}")
else:
    print(f"\n✅ FINAL VIDEO: {FINAL_VIDEO}")

print("\n" + "=" * 70)
print("POST-PRODUCTION COMPLETE")
print("=" * 70)
print(f"\nDelivery:")
print(f"  File: {FINAL_VIDEO if final_duration <= 180 else trimmed_video}")
print(f"  Duration: {min(final_duration, 180):.1f} seconds")
print(f"  Size: {final_size_mb:.1f} MB")
