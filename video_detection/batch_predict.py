import os
import argparse
from video_detection.video_predict import analyze_video  # <- fix here

def batch_process_videos(video_dir, mode="seen"):
    assert os.path.exists(video_dir), f"Video directory not found: {video_dir}"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"\n📂 Found {len(video_files)} videos in: {video_dir}\n{'='*50}")
    
    for video in video_files:
        video_path = os.path.join(video_dir, video)
        analyze_video(video_path, frame_interval=30, mode=mode)
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Path to video folder")
    parser.add_argument("--mode", type=str, default="seen", choices=["seen", "unseen"], help="Evaluation mode")
    args = parser.parse_args()

    batch_process_videos(args.video_dir, args.mode)
