from Advanced_Lane_Detect.ImageReader import read_images
from Advanced_Lane_Detect.Thresholding import threshold_pipeline
from Advanced_Lane_Detect.ImageTransformer import calibrate_camera, undistorter, apply_perspective_transform
#from Advanced_Lane_Detect.LineFinder import find_lanes
from Advanced_Lane_Detect.LineMaintainer import LineMaintainer
from moviepy.editor import VideoFileClip
import cv2
import matplotlib.pyplot as plt
import numpy as np

CAMERA_SAVED = True


images = read_images()
mtx, dist = None, None
if not CAMERA_SAVED:
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(display=False)
    np.save('../mtx.npy', mtx)
    np.save('../dist.npy', dist)
else:
    mtx = np.load('../mtx.npy')
    dist = np.load('../dist.npy')

left_line = LineMaintainer(line_dir='left')
right_line = LineMaintainer(line_dir='right')

def pipeline(image):
    undistored_image = undistorter(image, mtx, dist, display=False, saved=CAMERA_SAVED)
    thresholded_image = threshold_pipeline(undistored_image, display=False)
    perspective_image, Minv, M = apply_perspective_transform(thresholded_image, display=False, point_display=False)
    left_fitx, warped, ploty, left_fit, left_radius, left_detected = left_line.search_line(perspective_image, display_histogram=False, display_blindsearch_boxes=False)
    right_fitx, warped, ploty, right_fit, right_radius, right_detected = right_line.search_line(perspective_image, display_histogram=False, display_blindsearch_boxes=False)
    center = (left_fitx[-1] + right_fitx[-1])//2
    pos = (image.shape[1]//2 - center)*(3.7 / 700)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistored_image, 1, newwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Left radius of curvature  = %.2f m' % (left_radius), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Right radius of curvature = %.2f m' % (right_radius), (50, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle position : %.2f m %s of center' % (abs(pos), 'left' if pos < 0 else 'right'), (50, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result

output = 'result.mp4'
clip = VideoFileClip('../project_video.mp4')
white_clip = clip.fl_image(pipeline)
white_clip.write_videofile(output, audio=False)