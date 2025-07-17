# factor_module/factor_score.py

import numpy as np
import cv2
from insightface.app import FaceAnalysis
import onnxruntime as ort

# Initialize InsightFace with GPU (and CPU fallback)
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(image_path):
    """
    Extracts facial embedding from a given image using InsightFace.
    Returns None if no face is detected or image cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return None

    faces = app.get(img)
    if not faces:
        print(f"❌ No face detected in image: {image_path}")
        return None

    return faces[0].embedding

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    Returns 0.0 if either is None.
    """
    if vec1 is None or vec2 is None:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def factor_score(image_path, reference_embeddings_dict):
    """
    Calculates the highest cosine similarity between the input image
    and a set of reference embeddings.

    Args:
        image_path (str): Path to image.
        reference_embeddings_dict (dict): {identity_name: embedding}

    Returns:
        float: maximum cosine similarity
    """
    emb = get_embedding(image_path)
    if emb is None:
        return 0.0

    similarities = [
        cosine_similarity(emb, ref_emb)
        for ref_emb in reference_embeddings_dict.values()
    ]

    return max(similarities) if similarities else 0.0
