import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
from utils import euclidean_distance
from utils import trans_checker_stand_coor
from utils import transform_coordinate
from utils import draw_ar_points
from utils import measure_width_height
from utils import make_cheker_points


def draw(img, corners, imgpts):
    corner = tuple([int(corners.ravel()[0]), int(corners.ravel()[1])])
    img = cv2.line(
        img, corner, tuple(list(map(int, imgpts[0].ravel()))), (255, 0, 0), 5
    )
    img = cv2.line(
        img, corner, tuple(list(map(int, imgpts[1].ravel()))), (0, 255, 0), 5
    )
    img = cv2.line(
        img, corner, tuple(list(map(int, imgpts[2].ravel()))), (0, 0, 255), 5
    )
    return img


def make_cube_axis(checker_num, checker_size, cube_size):
    """
    큐브 그리기
    checker_num : 시작점 체커 번호 (int)
    checker_size : 내부 체커교차점 개수 (xline, yline) - ex) (7, 6)
    cube_size : (x, y, z) 형태 - ex) (3, 4, 5)
    return values : axis (xyz 축방향 3개 좌표), axisCube (8개 좌표)
    """

    xline_size = checker_size[0]
    yline = checker_num // xline_size
    xline_number = xline_size * yline
    x, y, z = cube_size

    axis = np.float32(
        [
            [x + checker_num - xline_number, yline, 0],  # 파란색
            [checker_num - xline_number, y + yline, 0],  # 초록색
            [checker_num - xline_number, yline, -z],  # 빨강색
        ]
    ).reshape(-1, 3)
    axisCube = np.float32(
        [
            [checker_num - xline_number, yline, 0],  # [0, 0, 0],
            [checker_num - xline_number, y + yline, 0],  # [0, 3, 0],
            [x + checker_num - xline_number, y + yline, 0],  # [3, 3, 0],
            [x + checker_num - xline_number, yline, 0],  # [3, 0, 0],
            [checker_num - xline_number, yline, -z],  # [0, 0, -3],
            [checker_num - xline_number, y + yline, -z],  # [0, 3, -3],
            [x + checker_num - xline_number, y + yline, -z],  # [3, 3, -3],
            [x + checker_num - xline_number, yline, -z],  # [3, 0, -3],
        ]
    )
    return axis, axisCube


# img = cv2.imread("checker1.jpg")
# pts1 = np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])

img = cv2.imread("img5.jpg")
h, w = img.shape[:2]

# 체커 보드 탐색 4개 좌표
pts1 = np.float32([[849, 1581], [733, 2435], [1721, 1576], [1792, 2426]])

# 물체 탐색 후 외곽 6개 좌표
# 빅파이
big_pai = [[760, 419], [804, 664], [1153, 1235], [2307, 807], [2374, 597], [1868, 132]]

# pts1의 x축 좌표간의 간격을 기준으로 나머지 4개의 좌표를 구해준다.
pts2 = trans_checker_stand_coor(pts1, (w, h * 2))
M = cv2.getPerspectiveTransform(pts1, pts2)

measure_width_height(pts1, big_pai, 4, M, (3, 3))
# draw_ar_points(img, pts1, 24, 2000)

ar_checker_points, checker_points = make_cheker_points(img, pts1, 24)

# [804, 664] 와 가장 가까운 체커 코너 index 구하기
dist_min = w
min_idx = 0
for idx, coor in enumerate(checker_points):
    dist = euclidean_distance(coor, [804, 664])
    if dist < dist_min:
        dist_min = dist
        min_idx = idx

print("index, dist_min", min_idx, dist_min)
print(len(checker_points))


# Load previously saved data
with np.load("checker_4_3.npz") as X:
    mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((216 * 294, 3), np.float32)
objp[:, :2] = np.mgrid[0:216, 0:294].T.reshape(-1, 2)


image = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# checker_points 형식을 맞춰주고
checker_points = np.array(checker_points)

corners2 = cv2.cornerSubPix(gray, checker_points, (11, 11), (-1, -1), criteria)

ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

# x, y축 길이 행렬 사이즈 알아야 되고
axis, axisCube = make_cube_axis(24620, (216, 294), (3, 3, 3))

imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

img = draw(img, checker_points[24620], imgpts)

cv2.resize(img, (w, h))
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

# for i in big_pai:
#     cv2.circle(img, tuple(list(map(int, i))), 10, (0, 0, 255), -1, cv2.LINE_AA)
# for i in pts1:
#     cv2.circle(img, tuple(list(map(int, i))), 10, (0, 0, 255), -1, cv2.LINE_AA)

# dst = cv2.warpPerspective(img, M, (w * 3, h * 3))

# print("빅파이 21 15 4.5")
# re_M = cv2.getPerspectiveTransform(pts2, pts1)

# # draw_ar_points(img, pts1, 3, 500)

# re_dst = cv2.warpPerspective(dst, re_M, (w, h))
# re_dst = cv2.resize(re_dst, (w // 4, h // 4))

# cv2.imshow("re_dst", re_dst)
# cv2.waitKey()
# cv2.destroyAllWindows()
