import glob

import numpy as np
import cv2
import cv2.aruco as aruco
import pickle

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=5,
    markersY=7,
    markerLength=10,
    markerSeparation=2,
    dictionary=aruco_dict)

cv2.imwrite(r"../data/Aruco_Markers/Calibration/Boards/Board.jpg", b := board.draw((500 + 80, 700 + 120), 10))

for iname in glob.glob(r"../data/Aruco_Markers/Calibration/Calibration_Image*.jpg"):
    print("\nImage > {}".format(iname))
    QueryImg = cv2.imread(iname, 1)

    gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print("# of IDs > {}".format(len(ids)))

    if ids is not None and corners is not None and len(ids) > 0 and len(corners) > 0 and len(corners) == len(ids):
        if len(ids) == len(board.ids):

            # Calibrate the camera now using cv2 method
            ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
                objectPoints=board.objPoints,
                imagePoints=corners,
                imageSize=gray.shape,  # [::-1], # may instead want to use gray.size
                cameraMatrix=None,
                distCoeffs=None)

            print(cameraMatrix)
            print(distCoeffs)

            # # Calibrate camera now using Aruco method
            # ret, cameraMatrix, distCoeffs, _, _ = aruco.calibrateCameraAruco(
            #     corners=corners,
            #     ids=ids,
            #     counter=35,
            #     board=board,
            #     imageSize=gray.shape[::-1],
            #     cameraMatrix=None,
            #     distCoeffs=None)
            #
            # print(cameraMatrix)
            # print(distCoeffs)

        with open(r"../data/Aruco_Markers/Calibration/Calibration_Images_Charuco/calibration.pckl", 'wb') as f:
            pickle.dump((cameraMatrix, distCoeffs), f)

        print('Calibration successful.')

cv2.destroyAllWindows()