import cv2
import numpy as np

image = cv2.imread('DM_sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

cv2.imshow('Original Image', image)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()