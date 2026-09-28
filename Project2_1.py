import json
import numpy as np
from volumeViewer import volumeViewer

# Load data
f = open('Project2.json', encoding='utf-8-sig')
dt = json.load(f)
f.close()

mr = np.array(dt['mr']['data'])
voxsz = np.array(dt['mr']['voxsz'])
lvr = np.array(dt['lvr']['data'])
msk1 = np.array(dt['msk1'])
msk2 = np.array(dt['msk2'])

# Create volume viewer
viewer = volumeViewer("Ground Truth Liver Segmentation")

# Display MRI volume
viewer.setImage(mr, voxsz)

# Overlay ground truth liver segmentation
viewer.addMask(
    lvr,
    color=[0, 1, 0],
    opacity=0.5,
    label="Ground Truth Liver"
)

# Display viewer
viewer.display()