import numpy as np
import cv2
import matplotlib.pyplot as plt
from Advanced_Lane_Detect.ImageReader import read_images

images = read_images(project=False, camera=True)
img_size = (images[0].shape[1], images[0].shape[0])

objpoints = []
imgpoints = []

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)

def calibrate_camera(display=False):
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(image, (9,6), corners, ret)
            if display:
                plt.imshow(img)
                plt.show()
    np.save('../objpoints.npy', objpoints)
    np.save('../imgpoints.npy', imgpoints)
    print("Camera Calibrated")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return ret, mtx, dist, rvecs, tvecs

def undistorter(image, mtx, dist, objpoints=objpoints, imgpoints=imgpoints, display=False, saved=False):
    if saved:
        objpoints = np.load('../objpoints.npy')
        imgpoints = np.load('../imgpoints.npy')

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise Exception("Calibrate Camera Before Undistorting")

    undist = cv2.undistort(image, mtx, dist, None, mtx)
    if display:
        plt.title("Undistorted Image")
        plt.imshow(undist)
        plt.show()
    return undist

def apply_perspective_transform(image, display=False, point_display=False):
    p1 = [550, 450]
    p2 = [720, 450]
    p3 = [1250, 720]
    p4 = [40, 720]
    dp1 = [0, 0]
    dp2 = [1280, 0]
    dp3 = [1250, 720]
    dp4 = [40, 720]
    if point_display:
        plt.imshow(image, cmap='gray')
        plt.plot(p1[0], p1[1], ".", color='r', markersize=20)
        plt.plot(p2[0], p2[1], ".", color='r', markersize=20)
        plt.plot(p3[0], p3[1], ".", color='r', markersize=20)
        plt.plot(p4[0], p4[1], ".", color='r', markersize=20)
        plt.plot(dp1[0], dp1[1], ".", color='b', markersize=12)
        plt.plot(dp2[0], dp2[1], ".", color='b', markersize=12)
        plt.plot(dp3[0], dp3[1], ".", color='b', markersize=12)
        plt.plot(dp4[0], dp4[1], ".", color='b', markersize=12)
        plt.show()

    img_size = (image.shape[1], image.shape[0])
    src = np.float32([p1, p2, p3, p4])
    dest = np.float32([dp1, dp2, dp3, dp4])
    perspective_transform = cv2.getPerspectiveTransform(src, dest)
    inverse_perspective_transform = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(image, perspective_transform, img_size)
    if display:
        plt.imshow(warped, cmap='gray')
        plt.show()

    return warped, inverse_perspective_transform, perspective_transform




