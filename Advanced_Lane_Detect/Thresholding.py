import numpy as np
import matplotlib.pyplot as plt

from Advanced_Lane_Detect.SobelThresholding import sobel_pipeline
from Advanced_Lane_Detect.ColorThresholding import color_pipeline

GRAD_X_THRESHOLD = (20, 100)
GRAD_Y_THRESHOLD = (60, 100)
MAG_THRESHOLD = (60, 100)
DIR_THRESHOLD = (0.8, 1.3)
LUV_THRESHOLD = (150, 255)
HSV_THESHOLD = (225, 255)
KSIZE = 15

def threshold_pipeline(image, display=False):
    img = np.copy(image)
    sobel_thresh_img = sobel_pipeline(image=img, ksize=KSIZE, GRAD_X_THRESHOLD=GRAD_X_THRESHOLD, GRAD_Y_THRESHOLD=GRAD_Y_THRESHOLD,
                               MAG_THRESHOLD=MAG_THRESHOLD, DIR_THRESHOLD=DIR_THRESHOLD, display=display)
    color_thresh_img = color_pipeline(image=img, LUV_THRESHOLD=LUV_THRESHOLD, HSV_THRESHOLD=HSV_THESHOLD, display=display)
    combined_binary = np.zeros_like(color_thresh_img)
    combined_binary[(sobel_thresh_img == 1) | (color_thresh_img == 1)] = 1

    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=40)

        ax2.imshow(combined_binary, cmap='gray')
        ax2.set_title('Combined Thresold Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return combined_binary
