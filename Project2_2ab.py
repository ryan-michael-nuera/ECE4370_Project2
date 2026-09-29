import json
import numpy as np
from skimage.filters import threshold_multiotsu
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
msk1 = np.array(dt['msk1'])
msk2 = np.array(dt['msk2'])

# -------------------------------------------------------
# Part 2(a): Threshold the MR image
# -------------------------------------------------------
# Multi-Otsu with 3 classes splits the image into:
#   dark background/air | medium-intensity tissue | bright tissue (incl. liver)
# The upper of the two thresholds separates the liver from the darker structures.
thresholds = threshold_multiotsu(mr, classes=3)
print("Multi-Otsu thresholds:", thresholds)

threshold = thresholds[1]
seg_thresh = mr > threshold
print("Chosen threshold:", threshold)

# -------------------------------------------------------
# Part 2(b): Dice similarity coefficient
# -------------------------------------------------------
dice_check = myDice(msk1, msk2)
print("Dice(msk1, msk2): %.4f  (expected 0.4882)" % dice_check)

dice_thresh = myDice(lvr, seg_thresh)
print("Dice(ground truth, threshold result): %.4f" % dice_thresh)

# -------------------------------------------------------
# Visualize thresholding result (Part 2a)
# -------------------------------------------------------
viewer = volumeViewer("Part 2a: Thresholded MRI (Press Esc to quit)")
viewer.setImage(mr, voxsz)
viewer.addMask(seg_thresh, color=[1, 0, 0], opacity=0.5, label="Threshold result")
viewer.display()
