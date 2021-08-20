import cv2
import PIL.Image as Image
import os

IMAGES_PATH = "./img/"

IMAGE_ROW = 2
IMAGE_COLUMN = 4
IMAGE_SAVE_PATH = './img/final.png'

height = 1280
width = 962


# Define image stitching function
def image_grid(image_names, IMAGE_SIZE_C, IMAGE_SIZE_R):
    """
    Function that creates a blank canvas and puts all the images
    onto it, one by one, depending upon the chosen rows, coloumns
    and each image's size.

    :param IMAGE_SIZE_R:
    :param IMAGE_SIZE_C:
    :param image_names: List of images to stitch
    :return: The stitched image saved to directory, and return None
    """

    # Create a new canvas on which all the images would be placed
    to_image = Image.new('RGBA', (IMAGE_COLUMN * IMAGE_SIZE_C, IMAGE_ROW * IMAGE_SIZE_R))

    # Loop through all pictures, paste each picture
    for y in range(0, IMAGE_ROW):
        for x in range(0, IMAGE_COLUMN):
            from_image = image_names[y * IMAGE_COLUMN + x]
            to_image.paste(Image.fromarray(from_image), (x * IMAGE_SIZE_C, y * IMAGE_SIZE_R))

    to_image.save("./img/final.png")
    return to_image


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(int(width / 3), 100),
              font_scale=3,
              font_thickness=8,
              text_color=(0, 0, 0),
              text_color_bg=(255, 255, 255)
              ):
    padding = 10
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + padding, y + text_h + padding), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return img
