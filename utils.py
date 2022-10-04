import numpy as np
import cv2
from math import dist
import matplotlib.pyplot as plt


def load_npz(npz_file):
    with np.load(npz_file) as X:
        mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]
    return mtx, dist

def draw_outer_pts(
    image: np.ndarray,
    points: np.ndarray,
    points_size=7,
    win_size=(800, 800),
    win_name="CV Window",
):
    """
    image: 최외곽 점들을 그릴 이미지
    points: 최외곽 점들
    """
    cv2.circle(image, tuple(map(int, points[0].tolist())), points_size, (0, 0, 255), -1)
    cv2.circle(image, tuple(map(int, points[1].tolist())), points_size, (0, 0, 255), -1)
    cv2.circle(image, tuple(map(int, points[2].tolist())), points_size, (0, 0, 255), -1)
    cv2.circle(image, tuple(map(int, points[3].tolist())), points_size, (0, 0, 255), -1)

    image = cv2.resize(image, win_size)
    cv2.imshow(win_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def euclidean_distance(point1: list, point2: list) -> float:
    """
    유클리드 거리 구하기
    """
    return dist(point1, point2)


def transform_coordinate(trans_coor: np.array, point: list) -> list:
    """
    좌표를 바꾸고자 하는 변환 행렬을 통과시켜주는 함수
    trans_coor : 변환 행렬 (3X3) - np.array
    point : 변환하고자 하는 좌표 - ex) [300, 500]
    result : 변환된 좌표
    """
    # 2 col -> 3 col -> 3 row 1 col
    re_point = point.copy()
    re_point.append(1)
    # re_point = np.array(re_point).reshape(3, -1).tolist()

    after = trans_coor @ re_point

    # 3 row -> 3 col
    after = after.reshape(-1)  # wx`, wy`, w
    # w로 나눠준다 -> x`, y`
    result = [(after[0] / after[2]), (after[1] / after[2])]

    return result


def trans_checker_stand_coor(point: list, stand_corr: tuple, checker_size: tuple) -> list:
    """
    ** 수정 필요 **
    이미지상의 4개의 좌표를 일정한 간격으로 펴서 4개의 좌표로 만들어주는 함수
    point : ex) np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
    stand_coor : 새로 만드는 좌표의 좌측 상단의 기준 좌표
    result : point와 같은 형식 

    - 실행 예제 -
    img = cv2.imread("checker1.jpg")
    h, w = img.shape[:2]
    pts1 = np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
    ucl = euclidean_distance(pts1[0], pts1[2])
    pts2 = trans_checker_stand_coor(pts1, (w, h * 2))
    """

    # x, y 비율과 똑같이 ar 이미지에 투시한다.
    # 첫번째 좌표를 기준으로 오른쪽에서 x, 아래쪽 좌표에서 y 간격(비율)을 구해준다.
    # 1칸당 거리 구하기
    one_step = abs(point[0][0] - point[2][0]) / (checker_size[0] - 1)

    # y_ucl = abs(point[0][1] - point[1][1])

    w, h = stand_corr
    result = np.float32(
        [[w, h], 
        [w, h + one_step * (checker_size[1] - 1)], 
        [w + one_step * ((checker_size[0] - 1)), h], 
        [w + one_step * ((checker_size[0] - 1)), h + one_step * (checker_size[1] - 1)],]
    )

    return result


def make_cheker_points(
    image: np.array, original_points: list, checker_interval: int, iterations=500
) -> list:
    """
    image : image mat - np.array
    original_points : 기준이 되는 정사각형 행태의 체커 좌표 4개
    checker_interval : 4좌표 안 체커 중 한쪽면의 체커 개수 (양쪽면 체커 개수가 동일하다는 가정)
    iterations : 체커를 만드는 횟수 기본값 500

    - 실행 방법 예제 -
    img = cv2.imread("checker1.jpg")
    pts1 = np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
    ar_points, orgin_points =def make_cheker_points(img, pts1, 4)
    """

    h, w = image.shape[:2]
    pts1 = np.float32(original_points)

    # pts1의 x축 좌표간의 유클리드 거리
    euclidean = euclidean_distance(pts1[0], pts1[2])

    # 좌표를 펴서 정방향 이미지 좌측 상단 시작 좌표
    pts2 = trans_checker_stand_coor(pts1, (w, h * 2))

    ar_points = pts2.tolist()
    # x, y 축을 담당할 list를 만든다
    ar_points_x = [ar_points[0], ar_points[2]]
    ar_points_xy = ar_points_x.copy()

    start_x = ar_points_x[0][0]
    start_y = ar_points_x[0][1]
    term = euclidean // checker_interval

    x_axis = (w * 3) // term
    y_axis = (h * 3) // term

    print("x, y:", x_axis, y_axis)

    # 4 사분면
    while start_y < (h * 3):
        if start_x < (w * 3):
            new_point = [
                start_x,
                start_y,
            ]
            if new_point not in ar_points_xy:
                ar_points_xy.append(new_point)
            start_x += term
        else:
            start_x = ar_points_x[0][0]
            start_y += term

    start_x = ar_points_x[0][0]
    start_y = ar_points_x[0][1]
    # 3 사분면
    while start_y < (h * 3):
        if start_x > 0:
            new_point = [
                start_x,
                start_y,
            ]
            if new_point not in ar_points_xy:
                ar_points_xy.append(new_point)
            start_x -= term
        else:
            start_x = ar_points_x[0][0]
            start_y += term
    start_x = ar_points_x[0][0]
    start_y = ar_points_x[0][1]
    # 2 사분면
    while start_y > 0:
        if start_x > 0:
            new_point = [
                start_x,
                start_y,
            ]
            if new_point not in ar_points_xy:
                ar_points_xy.append(new_point)
            start_x -= term
        else:
            start_x = ar_points_x[0][0]
            start_y -= term
    start_x = ar_points_x[0][0]
    start_y = ar_points_x[0][1]
    # 1 사분면
    while start_y > 0:
        if start_x < (w * 3):
            start_x += term
            new_point = [
                start_x,
                start_y,
            ]
            if new_point not in ar_points_xy:
                ar_points_xy.append(new_point)
        else:
            start_x = ar_points_x[0][0]
            start_y -= term

    re_M = cv2.getPerspectiveTransform(pts2, pts1)

    # 원래 이미지에 그릴 반환 리스트
    result = list()

    # 정방향 좌표를 다시 원래 이미지 좌표로 바꿔준다.
    for point in ar_points_xy:
        new_p = transform_coordinate(re_M, point)
        if new_p[0] > 0 and new_p[0] < w and new_p[1] > 0 and new_p[1] < h:
            result.append(new_p)

    return ar_points_xy, result


### 그려주는것 말고도 좌표만 출력해주는것 만들자 ###
def draw_ar_points(
    image: np.array, original_points: list, checker_interval: int, iterations=500
):
    """
    *** 수정 필요 ***
    원본이미지에 체커 좌표 4개(정사각형 형태)를 찍어주면 전체 이미지에 모두 체커 간격으로 표시해준다.
    image : image mat - np.array
    original_points : 기준이 되는 정사각형 행태의 체커 좌표 4개
    checker_interval : 4좌표 안 체커 중 한쪽면의 체커 개수 (양쪽면 체커 개수가 동일하다는 가정)
    iterations : 체커를 만드는 횟수 기본값 500

    - 실행 방법 예제 -
    img = cv2.imread("checker1.jpg")
    pts1 = np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
    draw_ar_points(img, pts1, 4)
    """
    h, w = image.shape[:2]

    pts1 = np.float32(original_points)

    # pts1의 x축 좌표간의 유클리드 거리
    euclidean = euclidean_distance(pts1[0], pts1[2])
    # print("euclidean", euclidean)

    # 좌표를 펴서 정방향 이미지 좌표로 만든것
    pts2 = trans_checker_stand_coor(pts1, (w, h * 2))

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (w * 3, h * 3))
    ar_points_xy, _ = make_cheker_points(image, pts1, checker_interval, iterations)

    for point in ar_points_xy:
        dst = cv2.circle(
            dst, tuple(list(map(int, point))), 10, (255, 0, 0), -1, cv2.LINE_AA
        )

    re_M = cv2.getPerspectiveTransform(pts2, pts1)
    re_dst = cv2.warpPerspective(dst, re_M, (w, h))

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(221), plt.imshow(image), plt.title(
        "Original", fontdict={"fontsize": 20}
    )
    plt.subplot(222), plt.imshow(dst), plt.title(
        "Perspective", fontdict={"fontsize": 20}
    )
    plt.subplot(223), plt.imshow(re_dst), plt.title("remap", fontdict={"fontsize": 20})
    plt.show()


def make_cube_axis(checker_num: int, checker_size: tuple, cube_size: tuple) -> list:
    """
    체커 번호를 시작점으로 큐브와 xyz축을 그리기 위한 실제 세계의 좌표 구하기
    checker_num : 시작점 체커 번호 (int)
    checker_size : 내부 체커교차점 개수 (xline, yline) - ex) (7, 6)
    cube_size : (x, y, z) 형태 - ex) (3, 4, 5)
    return values : axis (xyz 축방향 3개 좌표), axisCube (8개 좌표) - np.float32
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


def pixel_coordinates(
    camera_mtx: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, real_coor: tuple
) -> np.ndarray:
    """
    camera_mtx: npz에 있는 mtx
    rvecs: rotation 변환 행렬
    tvecs: translation 변환 행렬
    real_coor: 현실 좌표계의 좌표
    반환값인 pixel_coor: 이미지상에서의 좌표
    """

    # Rodgigues notation으로 된 회전변환 행렬을 3x3 회전변환 행렬로 변환
    rotation_mtx = cv2.Rodrigues(rvecs)

    translation_mtx = tvecs

    # np.hstack으로 회전변환과 병진변환 행렬을 합쳐 [R|t]행렬 생성
    # Rodrigues()를 쓰면 0번째로 회전변환 행렬, 1번째로 변환에 사용한 Jacobian행렬이 나오므로 0번째만 사용
    R_t = np.hstack((rotation_mtx[0], translation_mtx))

    # 실제 좌표를 3x1행렬로 변환
    real_coor = np.array(real_coor).reshape(-1, 1)

    # 실제 좌표 마지막 행에 1을 추가
    real_coor = np.vstack((real_coor, np.array([1])))

    # 이미지 좌표계에서의 픽셀 좌표 연산
    pixel_coor = camera_mtx @ R_t @ real_coor

    # 마지막 행을 1로 맞추기 위해 마지막 요소값으로 각 요소를 나눔
    pixel_coor /= pixel_coor[-1]
    return pixel_coor[:2]
