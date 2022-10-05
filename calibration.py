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
    images = glob.glob("./calibration_image_(8,5)_3cm/*.jpg")
    # images = glob.glob("./3cm Calibration/*.jpg")
    # images = glob.glob("./temp/*.jpg")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    # criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.00001)
    imgpoints = []
    objpoints = []

    i,j = (8, 5)

    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//3, h//3))

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










0.28563495562675
0.32216123792590384
0.5178627760983673
0.3092883737381872
0.35554479674380673
0.23183887735618275
0.19502164896138524
0.24188670227459724
0.3402439015476595
0.36964917466555125
0.3703677893496522
0.3494062634750341
0.31451334017108706
0.2692893009926641
0.11915068220464457
0.10336177938916428
0.08276727103112837
0.1281207567695702
0.28228407570368774
0.25019337415502696
0.23888623792974814
0.23981950358771642
0.22225851900973786
0.49238606341549274
0.49904753263949553
0.38256493560574156
0.22902770730704008
0.41756427547848196
0.5095827545138516
0.2926663035165059
0.5205023530359306
0.3812071759813982
0.1778036641926526
0.1693383246818522
0.2759866770994819
0.45035234427064214