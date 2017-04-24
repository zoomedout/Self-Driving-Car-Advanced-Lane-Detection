import matplotlib.image as mpimg
import os

def read_images(project=True, camera=False):
    IMAGES_PATH = '../test_images/' if project else '../camera_cal/'
    image_paths = os.listdir(IMAGES_PATH)
    images = []

    for path in image_paths:
        image = mpimg.imread(IMAGES_PATH + path)
        images.append(image)

    return images


