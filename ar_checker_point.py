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


def find_approx(image):
    # input = cv2.imread(img)
    # image = remove(input)
    # h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)   # (1, 1)을 홀수값 쌍으로 바꿀수 있음 3,3 5,5 7,7.... 조절해가며 contours 상자를 맞춰감
    # edged = cv2.Canny(blurred, 6, 7)              # 6, 7을 바꿀수 있음 조절해가며 contours 상자를 맞춰감

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)    # 아래의 dilate 또는 closed를 사용
    # dilate = cv2.dilate(blurred, kernel, iterations =2)
    dilate = cv2.erode(blurred, kernel, iterations =2)

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours는 튜플로 묶인 3차원 array로 출력
    print(len(contours))

    # Make a mask image
    mask = np.zeros(image.shape).astype(image.dtype)

    color = [255,255,255]
    filled_image = cv2.fillPoly(mask, contours, color)

    # approx 출력값 그려보기
    approx_list = list()
    for cnt in contours:
        for eps in np.arange(0.001, 0.2, 0.001):
            length = cv2.arcLength(cnt, True)
            epsilon = eps * length
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 6 and length > 1000:      # approx가 6 -> 꼭짓점의 갯수
                # cv2.drawContours(filled_image,[approx],0,(0,0,255),10)
                approx_list.append(approx)
                break


    approx_resize = np.reshape(approx_list, (-1, 6, 2))

    return approx_resize


def fix_approx(contours):
    # 최소 y좌표
    y_coors = np.min(contours, axis=0)[1]
    print("y_coor", y_coors)
    # 좌상단 좌표가 index 0 번 -> 반시계 방향으로 좌표가 돌아간다.
    contours = contours.tolist()
    while y_coors != contours[-1][1]:
        temp = contours.pop(0)
        contours.append(temp)
    return contours
###### 객체

img = cv2.imread("img15.jpg")
h, w = img.shape[:2]

image = img.copy()

# 배경 제거
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
re_bg = remove(img)

print(re_bg.shape)
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

# 물체 탐색 후 외곽 6개 좌표
# big_pai = [[760, 419], [804, 664], [1153, 1235], [2307, 807], [2374, 597], [1868, 132]]
big_pai = find_approx(re_bg)

# 체커보드가 들어있는 좌표를 삭제
for pai in big_pai:
    x_min, y_min = np.min(pai, axis=0)
    x_max, y_max = np.max(pai, axis=0)

    if not ((x_min < pts1[1][0] < x_max) and (y_min < pts1[1][1] < y_max)):
        big_pai = pai
        break
print("big_pai", big_pai)

big_pai = fix_approx(big_pai)

print("꼭지점 get")

# 체커보드 4개의 좌표를 기준으로 변환 좌표를 구한다
pts2 = utils.trans_checker_stand_coor(
    pts1, (w, h * 2)
)  # pts1의 x축 좌표간의 간격을 나머지 4개의 좌표를 구해준다.

M = cv2.getPerspectiveTransform(pts1, pts2)

# 물체의 외곽선 좌표를 기준으로 가로, 세로 길이 구하기
utils.measure_width_height(pts1, big_pai, 4, M, checker_sizes)  # (몇바이, 몇)
# utils.draw_ar_points(img, pts1, 10)

# 높이 구하기
# def measure_height(object_points: np.array, mat: np.array)

pts1 = pts1.tolist()
ar_start = utils.transform_coordinate(M, pts1[0])
ar_second = utils.transform_coordinate(M, pts1[2])

ar_object_standard_z = utils.transform_coordinate(M, big_pai[1])


# 두 점을 1으로 나눈 거리를 1로 기준
standard_ar_dist = abs(ar_start[0] - ar_second[0]) / (checker_sizes[0] - 1)  # (몇바이, 몇)

# x, y, z 값을 갖는다
ar_object_real_coor = [
    (ar_object_standard_z[0] - ar_start[0]) / standard_ar_dist,
    (ar_object_standard_z[1] - ar_start[1]) / standard_ar_dist,
    0,
]

z = utils.pixel_coordinates(mtx, rvecs, tvecs, ar_object_real_coor)

image = cv2.warpPerspective(img, M, (w * 3 , h *3))

result_z = 0
for i in np.arange(0, 10, 0.01):
    if (z[1] - big_pai[0][1]) < 0:
        break
    z = utils.pixel_coordinates(
        mtx, rvecs, tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], -i)
    )
    result_z = i

    img = cv2.circle(img, tuple(list(map(int, z[:2]))), 5, (0, 0, 255), -1, cv2.LINE_AA)

# 그리기
image = cv2.circle(
    image, tuple(list(map(int, ar_object_standard_z))), 10, (0, 0, 255), -1, cv2.LINE_AA
)
font = cv2.FONT_HERSHEY_SIMPLEX
point = tuple(map(int, (ar_object_standard_z[0] + 100, ar_object_standard_z[1] - 100)))

cv2.putText(image, f"{ar_object_real_coor}", point, font, 10, (0, 255, 0), 10)

# 높이 좌표랑 만나는 z 좌표
cv2.putText(img, f"{z}", (400, 400), font, 5, (0, 255, 0), 10)
# 높이 출력
cv2.putText(img, f"height: {result_z*4}", (200, 200), font, 5, (0, 255, 0), 10)
print("높이길이 :",result_z*4)


for idx, p in enumerate([ar_start, ar_second]):
    image = cv2.circle(
        image, tuple(list(map(int, p))), 30, (255, 0, 0), -1, cv2.LINE_AA
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    point = tuple(map(int, (p[0], p[1] - 20)))
    # print(point)
    cv2.putText(image, f"{idx}", point, font, 10, (255, 0, 0), 10)

# 가로세로 그리기
cv2.line(img,(big_pai[1]), (big_pai[2]), (0, 255, 0), 5, cv2.LINE_AA)
cv2.line(img,(big_pai[2]), (big_pai[3]), (255, 0, 0), 5, cv2.LINE_AA)

img = cv2.resize(img, (w // 4, h // 4))
image = cv2.resize(image, (w // 3, h // 3))

cv2.imshow("image", image)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

