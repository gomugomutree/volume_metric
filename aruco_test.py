import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

# rotate a markers corners by rvec and translate by tvec if given input is the size of a marker.
# In the markerworld the 4 markercorners are at (x,y) = (+- markersize/2, +- markersize/2)
# returns the rotated and translated corners to camera world and the rotation matrix
def rotate_marker_corners(rvec, markersize, tvec = None):

    mhalf = markersize / 2.0
    # convert rot vector to rot matrix both do: markerworld -> cam-world
    mrv, jacobian = cv2.Rodrigues(rvec)

    #in markerworld the corners are all in the xy-plane so z is zero at first
    X = mhalf * mrv[:,0] #rotate the x = mhalf
    Y = mhalf * mrv[:,1] #rotate the y = mhalf
    minusX = X * (-1)
    minusY = Y * (-1)

    # calculate 4 corners of the marker in camworld. corners are enumerated clockwise
    markercorners = []
    markercorners.append(minusX + Y) #was upper left in markerworld
    markercorners.append(X + Y) #was upper right in markerworld
    markercorners.append(X + minusY) #was lower right in markerworld
    markercorners.append(minusX + minusY) #was lower left in markerworld
    # if tvec given, move all by tvec
    if tvec is not None:
        C = tvec #center of marker in camworld
        for i, mc in enumerate(markercorners):
            markercorners[i] = C + mc #add tvec to each corner

    markercorners = np.array(markercorners,dtype=np.float32) # type needed when used as input to cv2
    return markercorners, mrv


# with np.load("cap_calibration.npz") as X:
#     print(X)
    # cameraMatrix, distCoeffs, _, _ = [X[i] for i in ('cameraMatrix', 'distCoeffs', 'rvecs', 'tvecs')]
with open('calibration.pckl', 'rb') as f:
    data = pickle.load(f)
    cameraMatrix, distCoeffs = data

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()

#ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000) original
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=2,
        markersY=2,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)


# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers( # cornerSubPix
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)   

        # print('corners', corners)
        # QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
        

    if ids is not None:
        try:
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.09, cameraMatrix, distCoeffs) # solvePnP                                       
            # QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.09)

            cornerCoordinates, _ = rotate_marker_corners(rvec, 0.09, tvec)
            reducedDimensionsto2D, _ = cv2.projectPoints(cornerCoordinates, rvec, tvec, cameraMatrix, distCoeffs)
            reducedDimensionsto2D = np.int32(reducedDimensionsto2D).reshape(-1, 2) # reshape list for better readability
            print(reducedDimensionsto2D) # debugging
            print("a")

            # Draw square of projected points on the livestream for debugging
            QueryImg =cv2.polylines(QueryImg, [np.int32(reducedDimensionsto2D)], True, (0, 0, 0), 10)
        
        except:
            # print("Deu merda segue o baile")
            pass

        cv2.imshow('QueryImage', QueryImg)
        cv2.waitKey(0)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




########  보정 pckl 파일 만들기 - 영상으로 ###########

# import numpy as np
# import cv2
# import cv2.aruco as aruco
# import pickle

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# # Creating a theoretical board we'll use to calculate marker positions
# board = aruco.GridBoard_create(
#     markersX=5,
#     markersY=5,
#     markerLength=60,
#     markerSeparation=15,
#     dictionary=aruco_dict,)

#     # markersX = 5
#     # markersY = 7               
#     # markerLength = 60       
#     # markerSeparation = 15    
#     # dictionaryId = '6x6_250'  
#     # margins = markerSeparation
#     # borderBits = 1


# # Read an image or a video to calibrate your camera
# # I'm using a video and waiting until my entire gridboard is seen before calibrating
# # The following code assumes you have a 5X7 Aruco gridboard to calibrate with



# cam = cv2.VideoCapture(0)

# while(cam.isOpened()):
#     # Capturing each frame of our video stream
#     ret, QueryImg = cam.read()
#     if ret == True:
#         # grayscale image
#         gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
#         parameters = aruco.DetectorParameters_create()
#         # Detect Aruco markers
#         corners, ids, rejectedIamgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
#         cv2.imshow('QueryImage', QueryImg)

#         # Make sure markers were detected before continuing
#         if ids is not None and corners is not None and len(ids) > 0 and len(corners) > 0 and len(corners) == len(ids):
#             # The next if makes sure we see all matrixes in our gridboard
#             # Calibrate the camera now using cv2 method
#             if len(ids) == len(board.ids):
#                 ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
#                         objectPoints=board.objPoints,
#                         imagePoints=corners,
#                         imageSize=gray.shape, #[::-1], # may instead want to use gray.size
#                         cameraMatrix=None,
#                         distCoeffs=None)

#                 # Print matrix and distortion coefficient to the console
#                 print(cameraMatrix)
#                 print(distCoeffs)

#                 # Output values to be used where matrix+dist is required
#                 np.savez("cap_calibration.npz", cameraMatrix, distCoeffs, rvecs, tvecs)

#                 # PRint to console our success
#                 print('Calibration successful.')

#                 break

#     # Exit at the end of the video on the EOF key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
