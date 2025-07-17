import sys
import os
import cv2
import math
import argparse
import numpy as np
from tqdm import tqdm

# Import InsightFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Distance to image center
def distance_to_center(bbox, image):
    x1, y1, x2, y2 = bbox
    im_cx, im_cy = image.shape[1] // 2, image.shape[0] // 2
    bbox_cx, bbox_cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return math.sqrt((im_cx - bbox_cx) ** 2 + (im_cy - bbox_cy) ** 2)

# Pick the face closest to center
def select_best_face(faces, image):
    return min(faces, key=lambda f: distance_to_center(f.bbox, image))

# Align + crop using 5-point landmarks
def align_and_crop(image, face):
    return face_align.norm_crop(image, face.kps)

# Process a single image
def process_image(image_path, input_root, out_root, face_analyzer):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return

        faces = face_analyzer.get(image)
        if faces:
            best_face = select_best_face(faces, image)
            aligned = align_and_crop(image, best_face)
        else:
            aligned = image  # If no face found, keep original

        # Save aligned image in corresponding path
        relative_path = os.path.relpath(image_path, input_root)
        save_path = os.path.join(out_root, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, aligned)

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

# ------------------ MAIN ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Detection and Alignment with InsightFace")
    parser.add_argument('--input_root', required=True, help='Path to extracted face frames')
    parser.add_argument('--out_root', required=True, help='Output path for aligned faces')
    args = parser.parse_args()

    # Initialize InsightFace FaceAnalysis
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Recursively process images
    all_files = []
    for root, _, files in os.walk(args.input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))

    for image_path in tqdm(all_files, desc="🔍 Aligning Faces"):
        process_image(image_path, args.input_root, args.out_root, face_analyzer)