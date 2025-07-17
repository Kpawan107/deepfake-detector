# preprocess_for_xception.py

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 299x299
    img = cv2.resize(img, (299, 299))

    # Normalize to [-1, 1] range (Xception expects this)
    img = img / 255.0
    img = (img - 0.5) / 0.5

    # Convert back to uint8 for saving (rescale to [0, 255])
    img = ((img + 1) / 2 * 255).astype(np.uint8)

    return img

def preprocess_all(input_dir, output_dir):
    print(f"🔍 Scanning aligned faces in: {input_dir}")
    total = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            processed = preprocess_image(input_path)
            if processed is not None:
                cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                total += 1

    print(f"✅ Preprocessing complete! Total images saved: {total}")
    print(f"📁 Output directory: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess aligned face images for XceptionNet inference")
    parser.add_argument('--input_dir', required=True, help="Path to aligned face images")
    parser.add_argument('--output_dir', required=True, help="Path to save preprocessed images")

    args = parser.parse_args()
    preprocess_all(args.input_dir, args.output_dir)