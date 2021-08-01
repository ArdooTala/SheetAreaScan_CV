from cv2 import aruco
import cv2


while True:
    try:
        if (_id := input("id > ")) == 'q':
            break
        _id = int(_id)
    except:
        continue

    x = aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_5X5_250), _id, 600, None, 1)
    cv2.imwrite(r"data/Aruco_Markers/DICT_5X5_250/ID_" + str(_id) + ".png", x)
    cv2.imshow("Kir", x)


