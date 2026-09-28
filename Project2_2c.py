import numpy as np
from scipy import ndimage
from skimage.morphology import ball, binary_opening

# -------------------------------------------------------
# Part 2(c): Opening + largest connected component
# -------------------------------------------------------

# Use the threshold you selected in Part 2(a)
threshold = 100      # <-- CHANGE/TUNE THIS VALUE

# Threshold the MR image
seg_thresh = mr > threshold

# 3D morphological opening
# Try radius 1, 2, 3, etc.
radius = 2           # <-- CHANGE/TUNE THIS VALUE
se = ball(radius)

seg_open = binary_opening(seg_thresh, footprint=se)

# -------------------------------------------------------
# Keep only the largest 3D connected component
# -------------------------------------------------------

labels, num_labels = ndimage.label(seg_open)

# Compute the size of every component.
# np.bincount gives the number of voxels with each label.
component_sizes = np.bincount(labels.ravel())

# Label 0 is background, so don't allow it to be selected.
component_sizes[0] = 0

# Find label of largest foreground component
largest_label = np.argmax(component_sizes)

# Create final binary liver mask
seg_liver = labels == largest_label

# -------------------------------------------------------
# Dice score
# -------------------------------------------------------

dice = myDice(lvr, seg_liver)

print("Threshold:", threshold)
print("Opening radius:", radius)
print("Dice coefficient:", dice)