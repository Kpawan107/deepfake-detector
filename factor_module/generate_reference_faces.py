import os
import cv2
import argparse
from tqdm import tqdm

def extract_reference_faces(input_dir, output_dir, frames_per_video=5):
    """
    Extracts N aligned reference face images from each real video.
    """

    print(f"🔍 Scanning real videos from: {input_dir}")
    for dataset in os.listdir(input_dir):
        real_dir = os.path.join(input_dir, dataset, "real")
        if not os.path.isdir(real_dir):
            continue

        for video_folder in os.listdir(real_dir):
            video_path = os.path.join(real_dir, video_folder)
            if not os.path.isdir(video_path):
                continue

            # List and sort frame files
            frames = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
            if len(frames) < frames_per_video:
                continue  # skip if not enough frames

            selected_frames = frames[:frames_per_video]

            # Output path
            save_path = os.path.join(output_dir, dataset, video_folder)
            os.makedirs(save_path, exist_ok=True)

            for i, fname in enumerate(selected_frames):
                src = os.path.join(video_path, fname)
                dst = os.path.join(save_path, f"ref_{i:02d}.jpg")
                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img)

    print(f"✅ Done. Reference faces saved under: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Path to aligned_faces directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save reference faces")
    parser.add_argument('--frames_per_video', type=int, default=5, help="Number of frames to extract per video")
    args = parser.parse_args()

    extract_reference_faces(args.input_dir, args.output_dir, args.frames_per_video)
