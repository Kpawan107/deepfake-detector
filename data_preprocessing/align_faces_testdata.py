import sys
import os
import cv2
import math
import argparse
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Compute distance from bbox center to image center
def distance_to_center(bbox, image):
    x1, y1, x2, y2 = bbox
    im_cx, im_cy = image.shape[1] // 2, image.shape[0] // 2
    bbox_cx, bbox_cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return ((im_cx - bbox_cx) ** 2 + (im_cy - bbox_cy) ** 2) ** 0.5

# Select most centered face
def select_best_face(faces, image):
    return min(faces, key=lambda f: distance_to_center(f.bbox, image))

# Align and crop the image using keypoints
def align_and_crop(image, face):
    return face_align.norm_crop(image, face.kps)

# Process one image
def process_image(image_path, input_root, output_root, analyzer):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return

        faces = analyzer.get(image)
        if faces:
            best_face = select_best_face(faces, image)
            aligned = align_and_crop(image, best_face)
        else:
            aligned = image  # No face found, fallback to original

        relative_path = os.path.relpath(image_path, input_root)
        save_path = os.path.join(output_root, relative_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, aligned)

    except Exception as e:
        print(f"❌ Failed: {image_path} → {e}")

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and crop face frames in test_data")
    parser.add_argument("--input_root", default="test_data/processed_frames", help="Path to processed frames")
    parser.add_argument("--out_root", default="test_data/aligned_faces", help="Where to save aligned faces")
    args = parser.parse_args()

    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Collect all images
    all_images = []
    for root, _, files in os.walk(args.input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))

    # Process with progress bar
    for img_path in tqdm(all_images, desc="🔍 Aligning Faces"):
        process_image(img_path, args.input_root, args.out_root, face_analyzer)