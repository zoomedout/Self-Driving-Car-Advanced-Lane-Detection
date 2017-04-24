import numpy as np
import cv2
import matplotlib.pyplot as plt

def color_pipeline(image, LUV_THRESHOLD, HSV_THRESHOLD, display=False):

    def hsv_thresh(img, thresh=(0, 255)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        V = hsv[:, :, 2]
        binary = np.zeros_like(V)
        binary[(V > thresh[0]) & (V <= thresh[1])] = 1
        return binary

    def luv_threshold(img, thresh=(0, 255)):
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        L = luv[:,:,1]
        binary = np.zeros_like(L)
        binary[(L > thresh[0]) & (L <= thresh[1])] = 1
        return binary

    hsv_binary = hsv_thresh(image, thresh=HSV_THRESHOLD)
    luv_binary = luv_threshold(image, thresh=LUV_THRESHOLD)

    combined = np.zeros_like(hsv_binary)
    combined[(luv_binary == 1) | (hsv_binary == 1)] = 1

    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined Color', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return combined


