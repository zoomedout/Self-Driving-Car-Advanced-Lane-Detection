from Advanced_Lane_Detect.BlindLineFinder import blind_search_left, blind_search_right
from Advanced_Lane_Detect.ConfidentLineFinder import confident_search_left, confident_search_right
from collections import deque
import numpy as np


class LineMaintainer():

    def __init__(self, line_dir):
        self.detected = False
        self.line_dir = line_dir
        self.current_fit = deque(maxlen=8)
        self.current_fit_single = None

    def search_line(self, img, display_histogram=False, display_blindsearch_boxes=False, display_confident_path=False):
        if not self.detected:
            return self.blind_search(img, display_histogram, display_blindsearch_boxes)
        else:
            return self.confident_search(img, display_confident_path)

    def confident_search(self, img, display_confident_path=False):
        if self.line_dir == 'left':
            left_fitx, img, ploty, left_fit, left_curverad, detected = confident_search_left(self, img, display_confident_path)
            if detected and not self.insanity_check(left_curverad, left_fit):
                self.detected = True
                self.current_fit.append(left_fit)
                self.current_fit_single = left_fit
            else:
                self.detected = False
            return left_fitx, img, ploty, left_fit, left_curverad, detected
        else:
            right_fitx, img, ploty, right_fit, right_curverad, detected = confident_search_right(self, img, display_confident_path)
            if detected and not self.insanity_check(right_curverad, right_fit):
                self.detected = True
                self.current_fit.append(right_fit)
                self.current_fit_single = right_fit
            else:
                self.detected = False
            return right_fitx, img, ploty, right_fit, right_curverad, detected

    def blind_search(self, img, display_histogram=False, display_blindsearch_boxes=False):
        if self.line_dir == 'left':
            left_fitx, img, ploty, left_fit, left_curverad, detected = blind_search_left(img, display_histogram, display_blindsearch_boxes)
            if detected:
                self.detected = True
                self.current_fit.append(left_fit)
                self.current_fit_single = left_fit
            else:
                self.detected = False
            return left_fitx, img, ploty, left_fit, left_curverad, detected
        else:
            right_fitx, img, ploty, right_fit, right_curverad, detected = blind_search_right(img, display_histogram, display_blindsearch_boxes)
            if detected:
                self.detected = True
                self.current_fit.append(right_fit)
                self.current_fit_single = right_fit
            else:
                self.detected = False
            return right_fitx, img, ploty, right_fit, right_curverad, detected

    def insanity_check(self, radius, fit):
        return self.radius_check(radius) and self.distance_check(fit)

    def radius_check(self, radius):
        return radius < 200 or radius > 1000

    def distance_check(self, fit):
        if self.current_fit_single is None:
            return True
        value = np.absolute(self.current_fit_single[1] - fit[1])
        return value > 0.5