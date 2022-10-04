import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle
import utils

# with np.load("cap_calibration.npz") as X:
#     print(X)
    # camera_matrix, distCoeffs, _, _ = [X[i] for i in ('camera_matrix', 'distCoeffs', 'rvecs', 'tvecs')]
# with open('calibration.pckl', 'rb') as f:
#     data = pickle.load(f)
#     camera_matrix, distCoeffs = data

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)

board = aruco.GridBoard_create(
        markersX=3, 
        markersY=3,
        markerLength=0.04,
        markerSeparation=0.04,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures

# cam = cv2.VideoCapture(0)

for i in range(1 , 13):
    QueryImg = cv2.imread(f"./charuco_image/hexagon_image{i}.jpg")
# while(cam.isOpened()):
    # Capturing each frame of our video stream
    # ret, QueryImg = cam.read()
    # if ret == True:
    if True:
        # grayscale image 
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        # Creating a theoretical board we'll use to calculate marker positions

        #ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000) original
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    if ids is not None:
        try:
            # print(corners)
            print(ids)
            _, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=board.objPoints,
            imagePoints=corners,
            imageSize=gray.shape, #[::-1], # may instead want to use gray.size
            cameraMatrix=None,
            distCoeffs=None)

            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.09, camera_matrix, dist) # solvePnP                                       
            # print("find rvecs, tvecs")
            # QueryImg = aruco.drawAxis(QueryImg, camera_matrix, dist, rvecs, tvecs, 0.09)

            # Draw square of projected points on the livestream for debugging
            # QueryImg =cv2.polylines(QueryImg, [np.int32(reducedDimensionsto2D)], True, (0, 0, 0), 10)
            # print("corners",corners)
            # print('ids', ids)
            # print('rvecs', rvecs)
            # print('tvecs', tvecs)
        
            ar_object_real_coor = [0.00, 0.00, 0]
            # # pixel_coordinates 
            # print("rvecs", rvecs, rvecs.shape) # (1, 1, 3)
            # print("tvecs", tvecs, tvecs.shape) # (1, 1, 3)
            
            rvecs = rvecs[1].reshape(3, 1) # (1, 1, 3) -> (3, 1)
            tvecs = tvecs[1].reshape(3, 1) # (1, 1, 3) -> (3, 1)
            # print('rvecs', rvecs)
            # print('tvecs', tvecs)

            height_pixel = utils.pixel_coordinates(camera_matrix, rvecs, tvecs, ar_object_real_coor)
            # print(height_pixel)
            # outer_points1 = list(map(lambda x: x.tolist(), corners[0]))
            # print(corners)
            # print(outer_points1)
            # a, b, c, d = outer_points1[0][0]
            # print(a, b, c, d)
            # outer_points1 = np.float32([a, d, b, c])
            # print("outer_points1", outer_points1)

            # x축
            for i in np.arange(0, 0.9, 0.01):
                # if (height_pixel[1] - object_vertexes[0][1]) < 0:
                #     break
                height_pixel = utils.pixel_coordinates(
                    camera_matrix, rvecs, tvecs, (ar_object_real_coor[0]+i, ar_object_real_coor[1], 0)
                )
                height = i
                QueryImg = cv2.circle(QueryImg, tuple(list(map(int, height_pixel[:2]))), 5, (0, 255, 0), -1, cv2.LINE_AA)
            # y축
            for i in np.arange(0, 0.9, 0.01):
                # if (height_pixel[1] - object_vertexes[0][1]) < 0:
                #     break
                height_pixel = utils.pixel_coordinates(
                    camera_matrix, rvecs, tvecs, (ar_object_real_coor[0], ar_object_real_coor[1]+i, 0)
                )
                height = i
                QueryImg = cv2.circle(QueryImg, tuple(list(map(int, height_pixel[:2]))), 5, (255, 0, 0), -1, cv2.LINE_AA)
            # z축
            for i in np.arange(0, 1.9, 0.01):
                # if (height_pixel[1] - object_vertexes[0][1]) < 0:
                #     break
                height_pixel = utils.pixel_coordinates(
                    camera_matrix, rvecs, tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], i)
                )
                height = i
            QueryImg = cv2.circle(QueryImg, tuple(list(map(int, height_pixel[:2]))), 5, (0, 0, 255), -1, cv2.LINE_AA)
        except:
            # print("Deu merda segue o baile")
            pass
        cv2.imshow('QueryImage', QueryImg)
        cv2.waitKey(0)
    else:
        print("do not find corners")

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




exit()

# ########  보정 pckl 파일 만들기 - 영상으로 ###########

import numpy as np
import cv2
import cv2.aruco as aruco
import pickle

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=2,
    markersY=2,
    markerLength=1,
    markerSeparation=15,
    dictionary=aruco_dict,)

    # markersX = 5
    # markersY = 7               
    # markerLength = 60       
    # markerSeparation = 15    
    # dictionaryId = '6x6_250'  
    # margins = markerSeparation
    # borderBits = 1


# # Read an image or a video to calibrate your camera
# # I'm using a video and waiting until my entire gridboard is seen before calibrating
# # The following code assumes you have a 5X7 Aruco gridboard to calibrate with


cam = cv2.VideoCapture(0)
# cam = cv2.imread("./aruco_image/aruco_img10.jpg")

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        # Detect Aruco markers
        corners, ids, rejectedIamgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                                                                                                                                                            
        cv2.imshow('QueryImage', QueryImg)

        # Make sure markers were detected before continuing
        if ids is not None and corners is not None and len(ids) > 0 and len(corners) > 0 and len(corners) == len(ids):
        #     # The next if makes sure we see all matrixes in our gridboard
        #     # Calibrate the camera now using cv2 method
            if len(ids) == len(board.ids):
                ret, camera_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objectPoints=board.objPoints,
                    imagePoints=corners,
                    imageSize=gray.shape, #[::-1], # may instead want to use gray.size
                    cameraMatrix=None,
                    distCoeffs=None)

            # Print matrix and distortion coefficient to the console
                print(camera_matrix)
                print(distCoeffs)

# data = camera_matrix, distCoeffs, rvecs, tvecs

# with open('aruco_calibration.p', 'wb') as f:
#     pickle.dump(data, f)

# Output values to be used where matrix+dist is required
# np.savez("cap_calibration.npz", camera_matrix, distCoeffs, rvecs, tvecs)

# PRint to console our success
                print('Calibration successful.')

# break

#     # Exit at the end of the video on the EOF key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
