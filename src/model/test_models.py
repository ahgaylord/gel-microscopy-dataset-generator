import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-seg.pt')  # load a pretrained model

frame = cv2.imread("../vid/gel_screenshot3.png")
# Convert frame to HSV color space
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define local vars for instance parameters
lower = np.array([0, 25, 25])
upper = np.array([255, 255, 255])

# Apply color thresholding to extract red regions
mask = cv2.inRange(frame_hsv, lower, upper)

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imwrite("../vid/gel_screenshot3_masked.png", mask)

results = model.track(source="../vid/SMITH-physics_of_fluids-dye_video.avi", stream=True, show=True)
results = model.predict("../vid/gel_screenshot3.png")
# results.plot()

# model.info()  # display model information
# model.predict('path/to/image.jpg')  # predict