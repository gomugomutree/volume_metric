import atexit
import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
import utils

###### 객체
img = cv2.imread("img5.jpg")
h, w = img.shape[:2]
image = img.copy()

# npz 파일 받아오기
npz_file = "calibration3.npz"

# 카메라 계수, 왜곡 계수
mtx, dist = utils.load_npz(npz_file)

# 코너 사이즈, 보정 코너들, 회전 벡터, 변환 벡터
checker_sizes, refined_corners, rvecs, tvecs = utils.search_checkerboard_size(
    img, mtx, dist
)

# 체커 보드 탐색 4개 좌표
pts1 = utils.outer_pts(refined_corners, checker_sizes)

# 물체 탐색 후 외곽 6개 좌표
big_pai = [[760, 419], [804, 664], [1153, 1235], [2307, 807], [2374, 597], [1868, 132]]


# # check 1, 2
# 21 15 4.5
# start = [849, 1581]
# second = [1721, 1576]

# # 박파이 좌측 하단 모서리
# object_start = [804, 664]
# object_z = [760, 419]

# 체커보드 4개의 좌표를 기준으로 변환 좌표를 구한다
pts2 = utils.trans_checker_stand_coor(
    pts1, (w, h * 2)
)  # pts1의 x축 좌표간의 간격을 나머지 4개의 좌표를 구해준다.

M = cv2.getPerspectiveTransform(pts1, pts2)

# 물체의 외곽선 좌표를 기준으로 가로, 세로 길이 구하기
utils.measure_width_height(pts1, big_pai, 4, M, checker_sizes)  # (몇바이, 몇)
# utils.draw_ar_points(img, pts1, 10)

pts1 = pts1.tolist()
ar_start = utils.transform_coordinate(M, pts1[0])
ar_second = utils.transform_coordinate(M, pts1[2])

ar_object_standard_z = utils.transform_coordinate(M, big_pai[1])

# 두 점을 1으로 나눈 거리를 1로 기준
standard_ar_dist = abs(ar_start[0] - ar_second[0]) / checker_sizes[0]  # (몇바이, 몇)

# x, y, z 값을 갖는다
ar_object_real_coor = [
    (ar_object_standard_z[0] - ar_start[0]) / standard_ar_dist,
    (ar_object_standard_z[1] - ar_start[1]) / standard_ar_dist,
    0,
]

z = utils.pixel_coordinates(mtx, rvecs, tvecs, ar_object_real_coor)

# ######

# for i in np.arange(0, 2, 0.2):
#     z = utils.pixel_coordinates(mtx, rvecs, tvecs, (0, 0, -i))
#     img = cv2.circle(img, tuple(list(map(int, z[:2]))), 5, (0, 0, 255), -1, cv2.LINE_AA)
# img = cv2.resize(img, (w // 4, h // 4))

# cv2.imshow("img", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# exit()

# ########


image = cv2.warpPerspective(img, M, (w * 3, h * 3))


result_z = 0
for i in np.arange(0, 10, 0.01):
    if (z[1] - big_pai[0][1]) < 0:
        break
    z = utils.pixel_coordinates(
        mtx, rvecs, tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], -i)
    )
    result_z = i

    img = cv2.circle(img, tuple(list(map(int, z[:2]))), 5, (0, 0, 255), -1, cv2.LINE_AA)


# img = cv2.circle(
#     img, tuple(list(map(int, big_pai[1]))), 30, (0, 0, 255), -1, cv2.LINE_AA
# )

# 그리기
# image = cv2.circle(
#     image, tuple(list(map(int, ar_object_standard_z))), 30, (0, 0, 255), -1, cv2.LINE_AA
# )
font = cv2.FONT_HERSHEY_SIMPLEX
point = tuple(map(int, (ar_object_standard_z[0] + 100, ar_object_standard_z[1] - 100)))

cv2.putText(image, f"{ar_object_real_coor}", point, font, 10, (0, 255, 0), 10)

# 높이 좌표랑 만나는 z 좌표
cv2.putText(img, f"{z}", (400, 400), font, 5, (0, 255, 0), 10)
# 높이 출력
cv2.putText(img, f"height: {result_z*4}", (200, 200), font, 5, (0, 255, 0), 10)


for idx, p in enumerate([ar_start, ar_second]):
    image = cv2.circle(
        image, tuple(list(map(int, p))), 30, (255, 0, 0), -1, cv2.LINE_AA
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    point = tuple(map(int, (p[0], p[1] - 20)))
    # print(point)
    cv2.putText(image, f"{idx}", point, font, 10, (255, 0, 0), 10)

img = cv2.resize(img, (w // 4, h // 4))
image = cv2.resize(image, (w // 3, h // 3))

cv2.imshow("image", image)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

