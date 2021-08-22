import cv2
import numpy as np

from utils import image_grid, width, height, draw_text, biggest_contour, draw_rectangle, reorder

PATH = "./img/kindle_agot2.jpg"


def doc_scan_pipeline(input=PATH, output="./img/scanned_doc.jpg"):
    img = cv2.imread(input)

    # 0. Convert given image from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    img = cv2.resize(img, (width, height))

    # 1. Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Add Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    # 3. Add Canny edge detection
    img_threshold = cv2.Canny(img_blur, 100, 200, L2gradient=True)

    # 3.1 Apply dilation
    kernel = np.ones((3, 3))
    img_threshold = cv2.dilate(img_threshold, kernel, iterations=2)

    # 4. Find all the contours
    img_contours = img.copy()
    img_big_contour = img.copy()
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5)

    # 5. Find the biggest contour
    biggest, maxArea = biggest_contour(contours)
    biggest = reorder(biggest)
    cv2.drawContours(image=img_big_contour, contours=biggest, contourIdx=-1, color=(0, 255, 0), thickness=10)

    # 5.1 Draw a rectangle, i.e., 4 lines connecting the 4 dots corresponding to the largest contour
    img_big_contour = draw_rectangle(img_big_contour, biggest, thickness=2)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # 6. Image Warp
    # 6.1 Calculate a 3x3 perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 6.2 Apply the perspective matrix to the image
    img_warp_coloured = cv2.warpPerspective(img, matrix, (width, height))

    # 7. Adaptive thresholding
    img_warp_gray = cv2.cvtColor(img_warp_coloured, cv2.COLOR_BGR2GRAY)
    img_adaptive_th = cv2.adaptiveThreshold(img_warp_gray, 255, 1, cv2.THRESH_BINARY, 5, 2)

    # 7.1 Apply median blurring to remove tiny speckles of noise
    img_adaptive_th = cv2.medianBlur(img_adaptive_th, 3)

    # Save the document to disk
    cv2.imwrite(output, img_adaptive_th)

    # Add labels to each image
    img = draw_text(img, "Original")
    img_gray = draw_text(img_gray, "Grayscale")
    img_blur = draw_text(img_blur, "Gaussian Blur", pos=(int(width / 4), 50))
    img_threshold = draw_text(img_threshold, "Canny Edge", pos=(int(width / 4), 50))
    img_contours = draw_text(img_contours, "Contours")
    img_big_contour = draw_text(img_big_contour, "Largest Contour", pos=(int(width / 7), 50))
    img_warp_coloured = draw_text(img_warp_coloured, "Warp", pos=(int(width / 3), 50))
    img_adaptive_th = draw_text(img_adaptive_th, "Adaptive Thresholding", pos=(int(width / 7), 50),
                                font_scale=2, font_thickness=6)

    blank_img = np.zeros((height, width, 3), dtype=np.uint8)
    image_list = [img, img_gray, img_blur, img_threshold, img_contours, img_big_contour, img_warp_coloured, img_adaptive_th]

    # Combine the images into a grid
    # image_grid returns PIL image, np.asarray() can be used to convert it back to cv2 compatible format
    grid = np.asarray(image_grid(image_list, width, height))


if __name__ == "__main__":
    doc_scan_pipeline(input="./img/kindle_agot2.jpg", output="./img/scanned_doc.jpg")
    # doc_scan_pipeline(input="./img/tnk_art.jpg", output="./img/tnk_scan.jpg")
