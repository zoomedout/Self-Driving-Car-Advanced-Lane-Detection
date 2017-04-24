import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_defaults(img, display=False):
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((img, img, img)) * 255
    midpoint = np.int(histogram.shape[0] / 2)
    nwindows = 9
    window_height = np.int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    minpix = 25
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    if display:
        print_histogram(histogram)
    return histogram, out_img, midpoint, nwindows, window_height, nonzero, nonzerox, nonzeroy, margin, minpix, ploty

def blind_search_left(img, display_histogram=False, display_blindsearch_boxes=False):
    histogram, out_img, midpoint, nwindows, window_height, nonzero, nonzerox, nonzeroy, margin, minpix, ploty = load_defaults(img, display_histogram)
    leftx_base = np.argmax(histogram[:midpoint])
    leftx_current = leftx_base
    left_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox <= win_xleft_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)

    left_curverad = calculate_radius(img, leftx, lefty)
    print("Blind")
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    if display_blindsearch_boxes:
        print_blind_search_boxes(out_img, left_fitx, ploty)

    return left_fitx, img, ploty, left_fit, left_curverad, np.sum(leftx) > 0

def blind_search_right(img, display_histogram=False, display_blindsearch_boxes=False):
    histogram, out_img, midpoint, nwindows, window_height, nonzero, nonzerox, nonzeroy, margin, minpix, ploty = load_defaults(img, display_histogram)

    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    rightx_current = rightx_base
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) * (
            nonzeroy <= win_xright_low)).nonzero()[0]
        right_lane_inds.append(good_right_inds)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    right_lane_inds = np.concatenate(right_lane_inds)
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    right_fit = np.polyfit(righty, rightx, 2)

    right_curverad = calculate_radius(img, rightx, righty)
    print("Blind")
    # Now our radius of curvature is in meters00
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if display_blindsearch_boxes:
        print_blind_search_boxes(out_img, right_fitx, ploty)

    return right_fitx, img, ploty, right_fit, right_curverad, np.sum(rightx) > 0

def calculate_radius(img, x, y):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    # Now our radius of curvature is in meters
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curverad

def print_blind_search_boxes(out_img, fitx, ploty):
    plt.imshow(out_img)
    plt.plot(fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

def print_histogram(histogram):
    plt.figure()
    plt.plot(histogram)
    plt.show()