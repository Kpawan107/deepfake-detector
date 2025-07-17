import gradio as gr
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os

from cnn_detector.xception_for_inference import xception
from factor_module.factor import compute_factor_score
from fusion.ensemble import fuse_predictions

# ========== DEVICE SETUP ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== LOAD CNN MODEL ==========
model_path = os.path.join("models", "xception_focal_best.pth")
model = xception()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
model = model.to(device)
model.eval()

# ========== IMAGE TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# ========== IMAGE PREDICTION ==========
def predict_image(img):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_output = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    factor_score = compute_factor_score(img)  # 0.0 to 1.0
    final_score = fuse_predictions(cnn_output[1], 1 - factor_score, 0.8, 0.2)

    label = "Fake" if final_score > 0.5 else "Real"
    confidence = f"{final_score:.2f}"
    return f"Prediction: {label} (Confidence: {confidence})"

# ========== VIDEO PREDICTION ==========
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    if not cap.isOpened():
        return "❌ Could not open video."

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frames.append(frame)
    cap.release()

    if frame_count == 0:
        return "❌ No frames found in video."

    sampling_interval = max(1, frame_count // 10)  # Max 10 frames
    sampled_frames = frames[::sampling_interval]
    print(f"📽️ Total frames: {frame_count}, Using: {len(sampled_frames)} sampled frames")

    scores = []
    for i, frame in enumerate(sampled_frames):
        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                cnn_output = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            factor_score = compute_factor_score(img)
            final_score = fuse_predictions(cnn_output[1], 1 - factor_score, 0.8, 0.2)
            scores.append(float(final_score))
        except Exception as e:
            print(f"⚠️ Frame {i} skipped due to error: {e}")

    if len(scores) == 0:
        return "❌ No valid frames for prediction."

    avg_score = np.mean(scores)
    label = "Fake" if avg_score > 0.5 else "Real"
    confidence = f"{avg_score:.2f}"
    return f"Prediction: {label} (Confidence: {confidence})"

# ========== GRADIO INTERFACE ==========
iface_image = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs="text",
    title="TrustFace"
)

iface_video = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload Video"),
    outputs="text",
    title="TrustFace"
)

gr.TabbedInterface(
    [iface_image, iface_video],
    ["Image Detection", "Video Detection"]
).launch()
