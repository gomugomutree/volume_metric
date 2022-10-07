import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
import utils

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

if __name__ == "__main__":
    # images = glob.glob("./calibration_image_(8,5)_3cm/*.jpg")
    # images = glob.glob("./3cm Calibration/*.jpg")
    images = glob.glob("./charuco_image/*.jpg")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    # criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.00001)
    imgpoints = []
    objpoints = []

    i,j = (8, 5)
    resize = 4
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//resize, h//resize))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (i, j), None)
        if ret == True:
            objp = np.zeros((i * j, 3), np.float32)
            objp[:, :2] = np.mgrid[0:i, 0:j].T.reshape(-1, 2)
            objpoints.append(objp)
            checker_sizes = (i, j)
            # print(checker_sizes)

            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            imgpoints.append(refined_corners)
            print("corner is  detected !!!")

            # cv2.drawChessboardCorners(img, (6, 4), refined_corners, ret)
            # img = cv2.resize(img, (400, 800))
            # cv2.imshow("img", img)
            # cv2.waitKey()
            
        else:
            print("corner is not detected......")


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
    
    
    print("total error: ", tot_error/len(objpoints))

    if ret == True:
        np.savez(f"checker_{checker_sizes}.npz", ret = ret, mtx = camera_matrix, dist = dist, rvecs = rvecs, tvecs = tvecs, outer_points=outer_points, checker_size=checker_sizes)









