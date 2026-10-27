import sys

import cv2
import numpy as np

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)

# An image is a grid of numbers. Make a black 48x48 grayscale image.
image = np.zeros((48, 48), dtype=np.uint8)
print("Image shape:", image.shape)
print("Pixel dtype:", image.dtype)
print("Pixel range:", image.min(), "to", image.max())