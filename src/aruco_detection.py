import pickle
import cv2.aruco as aruco
import cv2
import glob
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


marker_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
detection_params = cv2.aruco.DetectorParameters_create()

with open(r"../data/Aruco_Markers/Calibration/Calibration_Images_Charuco/calibration.pckl", 'rb') as f:
    cameraMatrix, distCoeffs, _, _ = pickle.load(f)

for iname in glob.glob(r"../data/Aruco_Markers/Sample_Images/*.jpg"):
    print('='*100)
    print("Image > {}".format(iname))
    QueryImg = cv2.imread(iname, 1)
    gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, marker_dict, parameters=detection_params)
    QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 255, 0))

    ax = plt.figure().add_subplot(projection='3d')

    if ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 15, cameraMatrix, distCoeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            print('_'*100)
            QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 20)
            print("TRANSLATION >\n{}".format(tvec))
            rot_matrix, jac = cv2.Rodrigues(rvec)
            print("\nROTATION >\n{}".format(rot_matrix))

            ax.scatter(*tvec[0], zdir='y')

    cv2.imshow('QueryImage', QueryImg)

    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1000)
    ax.set_zlim(-100, 100)
    plt.show()

    cv2.waitKey(0)

cv2.destroyAllWindows()
