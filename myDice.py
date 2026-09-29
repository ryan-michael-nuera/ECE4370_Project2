import numpy as np


def myDice(mask1, mask2):
    """
    Dice similarity coefficient between two n-dimensional binary masks.

    Dice = 2|A ∩ B| / (|A| + |B|)
    """
    # Treat any nonzero value as "inside" the mask
    A = mask1 > 0
    B = mask2 > 0

    intersection = np.sum(A & B)
    total = np.sum(A) + np.sum(B)

    # If both masks are empty they agree perfectly
    if total == 0:
        return 1.0

    return 2.0 * intersection / total
