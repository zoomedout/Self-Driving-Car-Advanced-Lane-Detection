import numpy as np
import cv2
import matplotlib.pyplot as plt

def sobel_pipeline(image, ksize, GRAD_X_THRESHOLD, GRAD_Y_THRESHOLD, MAG_THRESHOLD, DIR_THRESHOLD, display=False):

    def abs_sobel_thresh(img, sobel_kernel=5, orient='x', thresh=(0, 255)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 2]
        sobel = cv2.Sobel(H, cv2.CV_64F, 1 if orient == 'x' else 0, 0 if orient == 'x' else 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sbinary

    def mag_thresh(img, sobel_kernel=5, thresh=(0, 255)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 2]
        sobelx = cv2.Sobel(H, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(H, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 2]
        sobelx = np.absolute(cv2.Sobel(H, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        sobely = np.absolute(cv2.Sobel(H, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        angle = np.arctan2(sobely, sobelx)
        sxbinary = np.zeros_like(angle)
        sxbinary[(angle >= thresh[0]) & (angle <= thresh[1])] = 1
        return sxbinary

    gradx = abs_sobel_thresh(img=image, orient='x', sobel_kernel=ksize, thresh=GRAD_X_THRESHOLD)
    grady = abs_sobel_thresh(img=image, orient='y', sobel_kernel=ksize, thresh=GRAD_Y_THRESHOLD)
    mag_binary = mag_thresh(img=image, sobel_kernel=ksize, thresh=MAG_THRESHOLD)
    dir_binary = dir_threshold(img=image, sobel_kernel=ksize, thresh=DIR_THRESHOLD)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Thresholded Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return combined
