"""
script to run SAM model
"""
import os
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from setup import HOME
import time
import numpy as np
import json

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cpu')
MODEL_TYPE = 'vit_h'

print("Using device:", DEVICE)

sam = sam_model_registry[MODEL_TYPE](
    checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_NAME = 'test.jpg'
IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

start = time.time()

sam_result = mask_generator.generate(image_rgb)

end = time.time()
print("Time taken:", (end - start))

masks = [
    mask['segmentation']
    for mask
    in sorted(sam_result, key=lambda x: x['area'], reverse=True)
]

# Convert the mask to uint8 data type
mask_uint8 = (masks[0] * 255).astype(np.uint8)

# Threshold the mask to get a binary mask
_, thresholded_mask = cv2.threshold(mask_uint8, 1, 255, cv2.THRESH_BINARY)

# Create a 4-channel image with alpha channel
image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)

# Apply the inverse mask to set the background to transparent
image_rgba[:, :, 3] = thresholded_mask

# Save the image with transparent background as PNG
cv2.imwrite("masked_image_transparent.png", image_rgba)

cv2.imshow("image", image_rgba)
cv2.waitKey(0)
cv2.destroyAllWindows()
