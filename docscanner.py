import cv2
import numpy as np

from utils import image_grid, width, height, draw_text

PATH = "./img/tnk_art.jpg"

img = cv2.imread(PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

img = cv2.resize(img, (width, height))

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = draw_text(img, "Original")

# Add Gaussian blur
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_gray = draw_text(img_gray, "Grayscale")

# Add thresholding
img_threshold = cv2.Canny(img_blur, 100, 200)  # APPLY CANNY BLUR
img_blur = draw_text(img_blur, "Gaussian Blur")

blank_img = np.zeros((height, width, 3), dtype=np.uint8)
img_threshold = draw_text(img_threshold, "Threashold")

image_list = [img, img_gray, img_blur, img_threshold, blank_img, blank_img, blank_img, blank_img]

image_grid = np.asarray(image_grid(image_list, width, height))

cv2.imshow('Document Scan', image_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
