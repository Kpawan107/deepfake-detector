# fusion/ensemble.py

def fuse_predictions(cnn_score, factor_score, cnn_weight=0.8, factor_weight=0.2):
    """
    Fuse CNN and FACTOR scores to compute final fake probability score.

    Args:
        cnn_score (float): Probability score from CNN (0 to 1, higher = more fake).
        factor_score (float): Similarity score from FACTOR (0 to 1, higher = more real).
        cnn_weight (float): Weight for CNN score.
        factor_weight (float): Weight for FACTOR score.

    Returns:
        float: Final fake probability score (0 to 1)
    """
    # Ensure input types
    cnn_score = float(cnn_score)
    factor_score = float(factor_score)

    # Invert FACTOR score (higher similarity = more real → 1 - score = more fake)
    factor_fake_score = 1.0 - factor_score

    # Weighted combination
    combined_score = (cnn_weight * cnn_score) + (factor_weight * factor_fake_score)

    return combined_score
