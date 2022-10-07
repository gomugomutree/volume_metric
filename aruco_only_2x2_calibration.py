import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
import utils
import cv2.aruco as aruco


def find_checker_outer_points(refined_corners: np.array, checker_sizes: tuple, printer=False):
    """
    points: cv2.cornerSubPix()에 의해 생성된 점들
    size: 체스판의 크기
    """
    points = refined_corners
    size = checker_sizes
        
    outer_points =  np.float32(
        [
            points[0][0],
            points[size[0] * (size[1] - 1)][0],
            points[size[0] - 1][0],
            points[(size[0] * (size[1] - 1)) + (size[0] - 1)][0],
            ]
        )
    if printer:
            print(size[0], size[1])
            print("0th", points[0][0])
            print("1st", points[size[0] * (size[1] - 1)][0])
            print("2nd", points[size[0] - 1][0])
            print("3rd", points[(size[0] * (size[1] - 1)) + (size[0] - 1)][0])

    return outer_points

aruco_dict = aruco.DICT_4X4_1000

ARUCO_DICT = aruco.Dictionary_get(aruco_dict)
ARUCO_PARAMETERS = aruco.DetectorParameters_create()


"""
오직 4개의 aruco만을 사용한 2 x 2 형태로 놓여진 aruco 에서만 사용가능
"""
if __name__ == "__main__":

    images = glob.glob("./aruco_calibration_image/ca*.jpg")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    imgpoints = []
    objpoints = []
    i,j = (4, 4)
    resize = 1
    
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        img = cv2.resize(img, (w // resize, h // resize))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        corners = [ utils.sort_aruco_marker(cor[0]) for cor in corners]

        if len(corners)  == len(ids) == 4:
            corners = utils.sort_each_aruco_markers(corners, len(ids))

            corners = utils.change_aruco_to_checker_corners(corners)
            
            corners = np.array(corners).reshape(-1, 1, 2) # (24, 1, 2)
            corners = corners.astype('float32')

            objp = np.zeros((i * j, 3), np.float32)
            objp[:, :2] = np.mgrid[0:i, 0:j].T.reshape(-1, 2)
            objpoints.append(objp)
            checker_sizes = (i, j)

            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            # print(refined_corners)
            imgpoints.append(refined_corners)
            print("corner is  detected !!!")

        else:
            print("corner is not detected......")

    # print(imgpoints)
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # calibration error
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
        print(i+1, error)

    ret, rvecs, tvecs = cv2.solvePnP(objp, refined_corners, camera_matrix, dist)
    outer_points = find_checker_outer_points(refined_corners, checker_sizes)
    total = tot_error/len(objpoints)
    print("total error: ", tot_error/len(objpoints))

    if ret == True:
        np.savez(f"aruco_cs_{checker_sizes}_re_{resize}_er_{total: .2f}.npz", ret = ret, mtx = camera_matrix, dist = dist, rvecs = rvecs, tvecs = tvecs, outer_points=outer_points, checker_size=checker_sizes)


