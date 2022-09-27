import atexit
from turtle import ycor
import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
import utils
from rembg import remove
"""
빅파이 21 15 4.5
밀크카라멜 7.2  4.5 2
초코파이 24.5 21 6
홀스 10 2 2
아카시아껌 7.5 2.4 2
레모나 11.5 6.7 2.7
칸쵸 10(윗지름) 6(아랫지름)    11
초코파이 소 12 9 6
와클 15 4.8 9.5
포켓몬 볼 7(지름)
"""
def measure_height(img: np.array, pts1: np.array, object_vertexes: np.array, checker_sizes: tuple, mat: np.array, mtx: np.array, rvecs: np.array, tvecs: np.array) -> int:

    pts1 = pts1.tolist()
    ar_start = utils.transform_coordinate(mat, pts1[0])
    ar_second = utils.transform_coordinate(mat, pts1[2])

    ar_object_standard_z = utils.transform_coordinate(mat, object_vertexes[1])


    # 두 점을 1으로 나눈 거리를 1로 기준
    standard_ar_dist = abs(ar_start[0] - ar_second[0]) / (checker_sizes[0] - 1)  # (몇바이, 몇)

    # x, y, z 값을 갖는다
    ar_object_real_coor = [
        (ar_object_standard_z[0] - ar_start[0]) / standard_ar_dist,
        (ar_object_standard_z[1] - ar_start[1]) / standard_ar_dist,
        0,
    ]

    z_coordinate = utils.pixel_coordinates(mtx, rvecs, tvecs, ar_object_real_coor)

    real_z = 0
    for i in np.arange(0, 10, 0.01):
        if (z_coordinate[1] - object_vertexes[0][1]) < 0:
            break
        z_coordinate = utils.pixel_coordinates(
            mtx, rvecs, tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], -i)
        )
        real_z = i

        img = cv2.circle(img, tuple(list(map(int, z_coordinate[:2]))), 5, (0, 0, 255), -1, cv2.LINE_AA)

    return z_coordinate, real_z, ar_object_standard_z



img = cv2.imread("img15.jpg")
h, w = img.shape[:2]

image = img.copy()

# 배경 제거
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
re_bg = remove(img)

# npz 파일 받아오기
npz_file = "calibration3.npz"

# 카메라 계수, 왜곡 계수 -> 카메라 고정값
mtx, dist = utils.load_npz(npz_file)

# 코너 사이즈, 보정 코너들, 회전 벡터, 변환 벡터
checker_sizes, refined_corners, rvecs, tvecs = utils.search_checkerboard_size(
    re_bg, mtx, dist
)

# 체커 보드 탐색 4개 좌표
pts1 = utils.outer_pts(refined_corners, checker_sizes)

# 물체 탐색 후 외곽 6개 좌표들의 list
vertexes = utils.find_vertex(re_bg)

# 체커보드 좌표가 들어있는 물체 좌표를 삭제
object_vertexes =  utils.find_object_vertex(vertexes, pts1)

# 물체의 꼭지점들을 정렬
object_vertexes = utils.fix_vertex(object_vertexes)

print("꼭지점 get")

# 체커보드 4개의 좌표를 기준으로 변환 좌표를 구한다
pts2 = utils.trans_checker_stand_coor(
    pts1, (w, h * 2)
)  # pts1의 x축 좌표간의 간격을 나머지 4개의 좌표를 구해준다.

# 투시 행렬 구하기
M = cv2.getPerspectiveTransform(pts1, pts2)

# 물체의 외곽선 좌표를 기준으로 가로, 세로 길이 구하기
width, vertical = utils.measure_width_vertical(pts1, object_vertexes, 4, M, checker_sizes, False)  # (몇바이, 몇)

# 높이 구하기 함수
z_coordinate, real_z, ar_object_standard_z = measure_height(img, pts1, object_vertexes, checker_sizes, M, mtx, rvecs, tvecs)

# 그리기
image = cv2.circle(
    image, tuple(list(map(int, ar_object_standard_z))), 10, (0, 0, 255), -1, cv2.LINE_AA
)
font = cv2.FONT_HERSHEY_SIMPLEX
point = tuple(map(int, (ar_object_standard_z[0] + 100, ar_object_standard_z[1] - 100)))

# 높이 좌표랑 만나는 z 좌표
# cv2.putText(img, f"{z_coordinate}", (400, 400), font, 5, (0, 255, 0), 10)
# 가로, 세로, 높이 출력
print("가로길이 :",width)
print("세로길이 :",vertical)
print("높이길이 :",real_z*4)

# 가로세로 그리기
cv2.putText(img, f"width:{width: .2f} vertical: {vertical: .2f} height: {real_z*4}", (w//5, object_vertexes[2][1]+100), font, 3, (255, 0, 0), 10)
cv2.line(img,(object_vertexes[1]), (object_vertexes[2]), (0, 255, 0), 5, cv2.LINE_AA)
cv2.line(img,(object_vertexes[2]), (object_vertexes[3]), (255, 0, 0), 5, cv2.LINE_AA)

img = cv2.resize(img, (w // 4, h // 4))
# image = cv2.resize(image, (w // 3, h // 3))
# cv2.imshow("image", image)

cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()





# ar 화면쪽에서 체커 포인트 0, 2번 좌표들
# for idx, p in enumerate([ar_start, ar_second]):
#     image = cv2.circle(
#         image, tuple(list(map(int, p))), 30, (255, 0, 0), -1, cv2.LINE_AA
#     )
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     point = tuple(map(int, (p[0], p[1] - 20)))
#     # print(point)
#     cv2.putText(image, f"{idx}", point, font, 10, (255, 0, 0), 10)