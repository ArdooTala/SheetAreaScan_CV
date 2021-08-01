import numpy
import cv2
from cv2 import aruco
import pickle
import glob


CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=10,
    markerLength=8,
    dictionary=ARUCO_DICT)

cv2.imwrite(r"../data/Aruco_Markers/Calibration/Boards/Board.jpg",
            b := CHARUCO_BOARD.draw((500, 700), 10))
cv2.imshow("Kir", b)
cv2.waitKey(1)

corners_all = []
ids_all = []
image_size = None

images = glob.glob(r"../data/Aruco_Markers/Calibration/Calibration_Images_Charuco/*.jpg")

for iname in images:
    img = cv2.imread(iname, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(
        image=gray,
        dictionary=ARUCO_DICT)

    img = aruco.drawDetectedMarkers(
        image=img,
        corners=corners)

    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=CHARUCO_BOARD)

    if response > 15:
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        img = aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids)

        if not image_size:
            image_size = gray.shape[::-1]

        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1] / proportion), int(img.shape[0] / proportion)))
        cv2.imshow('Charuco board', img)
        cv2.waitKey(0)
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

cv2.destroyAllWindows()

if not image_size:
    print(
        "Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    exit()

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners=corners_all,
    charucoIds=ids_all,
    board=CHARUCO_BOARD,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None)

print(cameraMatrix)
print(distCoeffs)

with open(r"../data/Aruco_Markers/Calibration/Calibration_Images_Charuco/calibration.pckl", 'wb') as f:
    pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)

print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))