import cv2
import os
import pickle
import numpy as np
import argparse
from datetime import datetime
import json
import torch

from cnn_detector.predict import get_cnn_score
from factor_module.factor_score import factor_score
from fusion.ensemble import fuse


def analyze_video(video_path, frame_interval=30, mode="seen"):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    predictions = []
    cnn_scores = []
    factor_scores = []
    frame_pred_dict = {}

    frame_idx = 0
    evaluated_frame_idx = 0

    basename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n🎥 Analyzing video: {basename} | Mode: {mode}")
    print(f"Total frames: {frame_count} | FPS: {fps} | Interval: {frame_interval}")

    # Select embedding based on mode
    ref_path = 'models/ref_embedding.pkl' if mode == 'seen' else 'models/ref_embedding_unseen.pkl'
    with open(ref_path, 'rb') as f:
        ref_emb = pickle.load(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            temp_frame_path = f'temp_frame_{frame_idx}.jpg'
            cv2.imwrite(temp_frame_path, frame)

            cnn_score = get_cnn_score(temp_frame_path)
            factor_score_val = factor_score(temp_frame_path, ref_emb)
            final_pred = fuse(cnn_score, factor_score_val)

            predictions.append(final_pred)
            cnn_scores.append(cnn_score)
            factor_scores.append(factor_score_val)

            frame_pred_dict[f"frame_{frame_idx}"] = {
                "cnn": float(round(cnn_score, 4)),
                "factor": float(round(factor_score_val, 4)),
                "final": final_pred
            }

            print(f"🧠 Frame {frame_idx}: CNN={cnn_score:.4f} | FACTOR={factor_score_val:.4f} --> {final_pred}")

            os.remove(temp_frame_path)
            evaluated_frame_idx += 1

        frame_idx += 1

    cap.release()

    # Majority Voting
    fake_count = predictions.count("Fake")
    real_count = predictions.count("Real")
    final_label = "Fake" if fake_count > real_count else "Real"

    print(f"\n📊 Video Evaluation Completed:")
    print(f"Frames Evaluated: {evaluated_frame_idx}")
    print(f"Real Predictions: {real_count} | Fake Predictions: {fake_count}")
    print(f"🧾 Final Decision: *** {final_label} ***")

    # Save results
    os.makedirs("evaluation/video_results", exist_ok=True)
    txt_report = f"evaluation/video_results/{basename}_{mode}_{timestamp}.txt"
    json_report = f"evaluation/video_results/{basename}_{mode}_{timestamp}.json"

    with open(txt_report, "w", encoding="utf-8") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Final Decision: {final_label}\n")
        f.write(f"Frames Evaluated: {evaluated_frame_idx}\n")
        f.write(f"Real Predictions: {real_count}\n")
        f.write(f"Fake Predictions: {fake_count}\n\n")
        for i, (cnn, factor, pred) in enumerate(zip(cnn_scores, factor_scores, predictions)):
            f.write(f"Frame {i}: CNN={float(cnn):.4f}, FACTOR={float(factor):.4f} --> {pred}\n")

    with open(json_report, "w") as jf:
        json.dump(frame_pred_dict, jf, indent=2)

    print(f"\n✅ Reports saved:")
    print(f"   • Text  → {txt_report}")
    print(f"   • JSON  → {json_report}")
    return final_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake video detection using CNN + FACTOR fusion.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--interval", type=int, default=30, help="Frame sampling interval (default: 30)")
    parser.add_argument("--mode", type=str, default="seen", choices=["seen", "unseen"], help="Evaluation mode: seen or unseen")
    args = parser.parse_args()

    print("🧠 CNN Device:", "GPU" if torch.cuda.is_available() else "CPU")
    analyze_video(args.video_path, args.interval, args.mode)
