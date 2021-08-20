import cv2
import numpy as np

from utils import image_grid, width, height, draw_text, biggest_contour, draw_rectangle, reorder

PATH = "./img/tnk_art.jpg"

img = cv2.imread(PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

img = cv2.resize(img, (width, height))

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Add Gaussian blur
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

# Add thresholding
img_threshold = cv2.Canny(img_blur, 100, 200)

# Find all the contours
img_contours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image=img_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=10)

# Find the biggest contour
biggest, maxArea = biggest_contour(contours)
biggest = reorder(biggest)
cv2.drawContours(image=imgBigContour, contours=biggest, contourIdx=-1, color=(0, 255, 0), thickness=20)
imgBigContour = draw_rectangle(imgBigContour, biggest, 2)

blank_img = np.zeros((height, width, 3), dtype=np.uint8)
image_list = [img, img_gray, img_blur, img_threshold, img_contours, imgBigContour, blank_img, blank_img]

# Add labels to each image
img = draw_text(img, "Original")
img_gray = draw_text(img_gray, "Grayscale")
img_blur = draw_text(img_blur, "Gaussian Blur", pos=(int(width / 4), 50))
img_threshold = draw_text(img_threshold, "Canny Edge", pos=(int(width / 4), 50))
img_contours = draw_text(img_contours, "Contours")
imgBigContour = draw_text(imgBigContour, "Largest contour", pos=(int(width / 5), 50))

# Combine the images into a grid
image_grid = np.asarray(image_grid(image_list, width, height))
