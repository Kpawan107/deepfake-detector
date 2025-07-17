import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def extract_frames_from_video(video_path, save_folder, frames_per_video=20):
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"⚠️ Skipping empty video: {video_path}")
        return

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

def extract_all_frames(input_dir, output_root, frames_per_video=20):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.mp4', '.avi')):
                continue

            video_path = os.path.join(root, file)
            rel_path = os.path.relpath(video_path, input_dir)
            video_folder= os.path.splitext(rel_path)[0]  # removes .mp4
            save_folder = os.path.join(output_root, video_folder)

            extract_frames_from_video(video_path, save_folder, frames_per_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from test videos into structured processed_frames folder")
    parser.add_argument('--input_dir', default='test_data', help='Root input folder containing raw test videos')
    parser.add_argument('--output_dir', default='test_data/processed_frames', help='Where to save extracted frames')
    parser.add_argument('--frames_per_video', type=int, default=20, help='Number of frames to extract per video')
    args = parser.parse_args()

    extract_all_frames(args.input_dir, args.output_dir, args.frames_per_video)
