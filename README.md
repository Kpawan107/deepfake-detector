# 🕵‍♂ Deepfake Detection using XceptionNet + FACTOR Fusion

This project implements a hybrid Deepfake Detection framework that combines a CNN-based classifier (XceptionNet) with identity-aware semantic verification (FACTOR). It leverages both spatial texture analysis and reference embedding similarity for robust detection of synthetic media.

## 🔧 Features

- 📦 Aligned face extraction from videos/images using InsightFace  
- 🧠 CNN-based classification using XceptionNet with Focal Loss  
- 🧬 FACTOR-based identity consistency scoring (ResNet18 embeddings)  
- 🔗 Fusion module: weighted combination of CNN + FACTOR predictions  
- 📊 ROC, Confusion Matrix, Accuracy, F1-Score, Precision & Recall  
- 🖥 Gradio Interface for user-friendly web-based detection  

## 🗂 Dataset Support

- FaceForensics++  
- DFDC (DeepFake Detection Challenge)  
- Celeb-DF V2  
- Reference identity set of 419 public figures  

## 📁 Project Structure


deepfake-detection/
│
├── data/
│   ├── raw_videos/           # Original videos
│   ├── aligned_faces/        # Preprocessed aligned face images
│   ├── reference_embeddings/ # FACTOR reference embeddings
│   └── labels/               # Ground-truth labels
│
├── models/
│   └── xception_focal_best.pth
│
├── scripts/
│   ├── extract_frames.py
│   ├── detect_and_align.py
│   ├── train_xception.py
│   ├── extract_feature.py
│   └── eval.py
│
├── gradio_app.py
└── README.md


## 🚀 Inference Demo

Run the Gradio interface to test:

bash
python gradio_app.py


Upload a video or image and receive Real/Fake prediction with confidence score.

## 🧪 Results (Summary)

| Model               | Accuracy | F1-Score |
| ------------------- | -------- | -------- |
| CNN Only            | 72.00%   | 71.50%   |
| FACTOR Only         | 68.00%   | 67.20%   |
| CNN + FACTOR Fusion | 78.50%   | 76.80%   |

- *ROC-AUC*: 87.38%  
- *Confusion Matrix*: True Real = 145, True Fake = 142  

## 📚 References

- [Detecting Deepfakes Without Seeing Any (CVPR 2023)](https://arxiv.org/abs/2304.07460)  
- [XceptionNet Paper](https://arxiv.org/abs/1610.02357)  
- [InsightFace GitHub](https://github.com/deepinsight/insightface)  
- [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo)