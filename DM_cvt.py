import cv2
import numpy as np

image = cv2.imread('DM_sample1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# 결과 이미지 저장 (추가)
cv2.imwrite('result1.jpg', depth_map)

cv2.imshow('Original Image', image)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()