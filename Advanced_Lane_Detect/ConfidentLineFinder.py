import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_defaults(img):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    return nonzerox, nonzeroy, margin, ploty, y_eval, out_img, window_img

def confident_search_left(LineObject, img, display_confident_path=False):
    nonzerox, nonzeroy, margin, ploty, y_eval, out_img, window_img = load_defaults(img)

    last_left_fit = np.array(LineObject.current_fit)
    first_coeff_mean = np.mean(last_left_fit[:, 0])
    second_coeff_mean = np.mean(last_left_fit[:, 1])
    third_coeff_mean = np.mean(last_left_fit[:, 2])
    left_lane_inds = ((nonzerox > (first_coeff_mean * (nonzeroy ** 2) + second_coeff_mean * nonzeroy + third_coeff_mean - margin)) & (
        nonzerox < (first_coeff_mean * (nonzeroy ** 2) + second_coeff_mean * nonzeroy + third_coeff_mean + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    left_curverad = calculate_radius(leftx, lefty, y_eval)
    # Now our radius of curvature is in meters
    print("Confident")

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    if display_confident_path:
        print_path(window_img, left_line_pts, out_img, left_fitx, ploty)

    return left_fitx, img, ploty, left_fit, left_curverad, np.sum(leftx) > 0

def confident_search_right(LineObject, img, display_confident_path=False):
    nonzerox, nonzeroy, margin, ploty, y_eval, out_img, window_img = load_defaults(img)
    last_right_fit = np.array(LineObject.current_fit)
    first_coeff_mean = np.mean(last_right_fit[:, 0])
    second_coeff_mean = np.mean(last_right_fit[:, 1])
    third_coeff_mean = np.mean(last_right_fit[:, 2])
    right_lane_inds = ((nonzerox > (first_coeff_mean * (nonzeroy ** 2) + second_coeff_mean * nonzeroy + third_coeff_mean - margin)) & (
        nonzerox < (first_coeff_mean * (nonzeroy ** 2) + second_coeff_mean * nonzeroy + third_coeff_mean + margin)))

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Now our radius of curvature is in meters
    right_curverad = calculate_radius(rightx, righty, y_eval)
    print("Confident")

    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    if display_confident_path:
        print_path(window_img, right_line_pts, out_img, right_fitx, ploty)

    return right_fitx, img, ploty, right_fit, right_curverad, np.sum(rightx) > 0


def calculate_radius(x,y,y_eval):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Now our radius of curvature is in meters
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * fit_cr[0])
    return left_curverad

def print_path(window_img, left_line_pts, out_img, left_fitx, ploty):
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)