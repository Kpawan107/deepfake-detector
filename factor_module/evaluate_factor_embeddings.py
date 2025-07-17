import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

def load_data(reference_path, test_path, labels_path):
    reference_embeddings = np.load(reference_path)
    test_embeddings = np.load(test_path)
    test_labels = np.load(labels_path)

    return reference_embeddings, test_embeddings, test_labels

def match_with_closest_reference(test_embeddings, reference_embeddings):
    sim_matrix = cosine_similarity(test_embeddings, reference_embeddings)
    max_sim_indices = np.argmax(sim_matrix, axis=1)
    max_sim_scores = np.max(sim_matrix, axis=1)
    return max_sim_indices, max_sim_scores

def evaluate(reference_path, test_path, labels_path):
    print("📥 Loading embeddings...")
    ref_embeds, test_embeds, test_labels = load_data(reference_path, test_path, labels_path)

    print("🔍 Matching test embeddings with reference identities...")
    _, sim_scores = match_with_closest_reference(test_embeds, ref_embeds)

    # Convert similarity to "fake probability"
    # Assume: high similarity = real, low similarity = fake
    probs = sim_scores  # already in [0, 1] due to cosine similarity

    print("📊 Evaluating...")
    auc = roc_auc_score(test_labels, probs)
    ap = average_precision_score(test_labels, probs)

    pred_labels = (probs >= 0.5).astype(int)

    print("\n✅ Classification Report:")
    print(classification_report(test_labels, pred_labels, target_names=["Fake", "Real"]))

    print("📈 Confusion Matrix:")
    print(confusion_matrix(test_labels, pred_labels))

    print(f"🎯 AUC: {auc * 100:.2f}")
    print(f"🎯 AP:  {ap * 100:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", type=str, required=True, help="Path to reference_embeddings.npy")
    parser.add_argument("--test_root", type=str, required=True, help="Path to test.npy")
    parser.add_argument("--labels_root", type=str, required=True, help="Path to labels.npy")

    args = parser.parse_args()
    evaluate(args.real_root, args.test_root, args.labels_root)
