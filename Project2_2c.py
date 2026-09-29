import json
import numpy as np
from scipy import ndimage
from skimage.morphology import ball, binary_opening
from volumeViewer import volumeViewer
from myDice import myDice

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
f = open('Project2.json', encoding='utf-8-sig')
dt = json.load(f)
f.close()

mr = np.array(dt['mr']['data'])
voxsz = np.array(dt['mr']['voxsz'])
lvr = np.array(dt['lvr']['data'])

# -------------------------------------------------------
# Part 2(c): Opening + largest connected component
# -------------------------------------------------------

# Threshold selected in Part 2(a) (upper multi-Otsu threshold)
threshold = 875

# Threshold the MR image
seg_thresh = mr > threshold

# 3D morphological opening with a spherical structuring element
radius = 3
se = ball(radius)
seg_open = binary_opening(seg_thresh, footprint=se)

# -------------------------------------------------------
# Keep only the largest 3D connected component
# -------------------------------------------------------
labels, num_labels = ndimage.label(seg_open)

# Number of voxels in each component (label 0 is background)
component_sizes = np.bincount(labels.ravel())
component_sizes[0] = 0

largest_label = np.argmax(component_sizes)
seg_liver = labels == largest_label

# -------------------------------------------------------
# Dice score
# -------------------------------------------------------
dice = myDice(lvr, seg_liver)

print("Threshold:", threshold)
print("Opening radius:", radius)
print("Connected components after opening:", num_labels)
print("Dice coefficient: %.4f" % dice)

# -------------------------------------------------------
# Visualize result with ground truth
# -------------------------------------------------------
viewer = volumeViewer("Part 2c: Filtered Segmentation vs Ground Truth (Press Esc to quit)")
viewer.setImage(mr, voxsz)
viewer.addMask(lvr, color=[0, 1, 0], opacity=0.5, label="Ground truth")
viewer.addMask(seg_liver, color=[1, 0, 0], opacity=0.5, label="Segmentation")
viewer.display()
