import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def extract_frames_from_video(video_path, save_folder, frames_per_video=32):
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < frames_per_video:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

    idx = 0
    saved = 0
    pbar = tqdm(total=len(frame_indices), desc=f"Extracting {os.path.basename(video_path)}", leave=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx in frame_indices:
            frame_filename = os.path.join(save_folder, f'frame_{saved:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved += 1
            pbar.update(1)

        idx += 1

    pbar.close()
    cap.release()

def extract_all_frames(input_dir, output_dir, frames_per_video=32):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.mp4', '.avi')):
                continue

            video_path = os.path.join(root, file)
            relative_path = os.path.relpath(video_path, input_dir)
            video_folder = os.path.splitext(relative_path)[0]  # remove extension

            save_folder = os.path.join(output_dir, video_folder)
            extract_frames_from_video(video_path, save_folder, frames_per_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from deepfake dataset while preserving structure")
    parser.add_argument('--input_dir', required=True, help='Path to input dataset (e.g., data/)')
    parser.add_argument('--output_dir', required=True, help='Path to save extracted frames (e.g., data/processed_frames/)')
    parser.add_argument('--frames_per_video', type=int, default=32, help='Number of frames to extract per video')
    args = parser.parse_args()

    extract_all_frames(args.input_dir, args.output_dir, args.frames_per_video)