import numpy as np
import cv2
from math import dist
import matplotlib.pyplot as plt

"""
    DESCRIPTIONS
    search_checkerboard_size: 
    outer_pts: 체커보드 이미지에서 최외곽 4개 점의 좌표를 계산하고 출력합니다.
    draw_outer_pts: 체커보드 이미지 상에 최외곽 점들을 도시할 수 있습니다.
"""


def load_npz(npz_file):
    with np.load(npz_file) as X:
        mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]
    return mtx, dist


def search_checkerboard_size(image: np.ndarray, mtx: np.ndarray, dist: np.ndarray):
    # image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    checker_sizes = []
    for i in range(7, 2, -1):
        a = 0
        for j in range(7, 2, -1):
            ret, corners = cv2.findChessboardCorners(gray, (i, j), None)
            if ret == True:
                objp = np.zeros((i * j, 3), np.float32)
                objp[:, :2] = np.mgrid[0:i, 0:j].T.reshape(-1, 2)
                check_size = (i, j)
                checker_sizes.append(check_size)
                print(checker_sizes)

                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                imgpoints.append(refined_corners)

                img = cv2.drawChessboardCorners(image, (i, j), refined_corners, ret)
                a = 1
                
                print("corner is detected!!")
                break
        if a == 1:
            break
    ret, rvecs, tvecs = cv2.solvePnP(objp, refined_corners, mtx, dist)
    return checker_sizes[-1], refined_corners, rvecs, tvecs


def outer_pts(points: np.ndarray, size: tuple) -> np.ndarray:
    """
    points: cv2.cornerSubPix()에 의해 생성된 점들
    size: 체스판의 크기
    """
    print(size[0], size[1])
    print("0th", points[0][0])
    print("1st", points[size[0] * (size[1] - 1)][0])
    print("2nd", points[size[0] - 1][0])
    print("3rd", points[(size[0] * (size[1] - 1)) + (size[0] - 1)][0])

    return np.float32(
        [
            points[0][0],
            points[size[0] * (size[1] - 1)][0],
            points[size[0] - 1][0],
            points[(size[0] * (size[1] - 1)) + (size[0] - 1)][0],
        ]
    )


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

    - 실행 예제 -
    img = cv2.imread("checker1.jpg")
    pts1 = np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
    h, w = img.shape[:2]
    ucl = euclidean_distance(pts1[0], pts1[2])
    pts2 = trans_checker_stand_coor(pts1, (w, h * 2))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    point = [931, 1411]
    re_point = transform_coordinate(M, point)
    print(pts2)
    pirnt(point)
    """

    # 2 col -> 3 col -> 3 row 1 col
    re_point = point.copy()
    re_point.append(1)
    re_point = np.array(re_point).reshape(3, -1).tolist()

    after = trans_coor @ re_point

    # 3 row -> 3 col
    after = after.reshape(-1)  # wx`, wy`, w
    # w로 나눠준다 -> x`, y`
    result = [(after[0] / after[2]), (after[1] / after[2])]

    return result


# 4개가 정사각형이라는 전제 하에서 작성한 함수
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


def measure_width_vertical(
    checker_points: list,
    object_points: list,
    check_real_dist: int,
    mat: np.array,
    checker_size: tuple,
    printer=True,
) -> list:
    """
    checker_points : checker board 기준 4개 좌표
    object_points : 물체의 외곽선 좌표 
    check_real_dist : 체크 1칸의 실제 거리 (cm)
    mat : 변환 행렬 (3 X 3) - cv2.getPerspectiveTransform(pts1, pts2)한 결과값
    printer : 가로, 세로 길이 출력문 실행 여부 - bool
    result : [가로 길이, 세로 길이] 

    - 실행 예제 -

    """
    re_point = list()
    checker_points = checker_points.tolist()
    # 체커보드가 정방향으로 투시되었을때 각 좌표들을 다시 구해준다.
    for point in checker_points:
        re_point.append(transform_coordinate(mat, point))

    re_object_points = list()
    for point in object_points:
        re_object_points.append(transform_coordinate(mat, point))

    # pt2[0]의 x축과 pt2[2]의 x축의 필셀 거리 // 코너 사이즈 - 1 (칸) = 1칸당 떨어진 픽셀거리
    one_checker_per_pix_dis = abs(re_point[0][0] - re_point[2][0]) / (
        checker_size[0] - 1
    )

    # 픽셀당 실제 거리 - check_real_dist(cm) / 1칸당 떨어진 픽셀 거리
    pix_per_real_dist = check_real_dist / one_checker_per_pix_dis

    # 두 점 사이의 픽셀거리 * 1픽셀당 실제 거리 = 두 점의 실제 거리
    width = (
        euclidean_distance(re_object_points[1], re_object_points[2]) * pix_per_real_dist
    )
    vertical = (
        euclidean_distance(re_object_points[2], re_object_points[3]) * pix_per_real_dist
    )

    if printer:
        print("1칸당 픽셀거리 :", one_checker_per_pix_dis)
        print("픽셀당 실제 거리 :", pix_per_real_dist)
        print("가로길이 :", width)
        print("세로길이 :", vertical)

    return [width, vertical]


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

def find_vertex(image: np.array) -> list:
    '''
    물체 꼭지점 6좌표 추출하는 함수
    iamge : 꼭지점을 찾을 이미지

    output -> vertex_resize : 꼭지점 좌표가 6개인 물체들의 좌표 리스트
    '''

    # h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)   # (1, 1)을 홀수값 쌍으로 바꿀수 있음 3,3 5,5 7,7.... 조절해가며 contours 상자를 맞춰감

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.erode(blurred, kernel, iterations =2)

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours는 튜플로 묶인 3차원 array로 출력
    # print(len(contours))

    # Make a mask image
    # mask = np.zeros(image.shape).astype(image.dtype)
    # color = [255,255,255]
    # filled_image = cv2.fillPoly(mask, contours, color)

    # vertex 출력값 그려보기
    vertex_list = list()
    for cnt in contours:
        for eps in np.arange(0.001, 0.2, 0.001):
            length = cv2.arcLength(cnt, True)
            epsilon = eps * length
            vertex = cv2.approxPolyDP(cnt, epsilon, True)
            if len(vertex) == 6 and length > 1000:      # vertex가 6 -> 꼭짓점의 갯수
                # cv2.drawContours(filled_image,[vertex],0,(0,0,255),10)
                vertex_list.append(vertex)
                break

    vertex_resize = np.reshape(vertex_list, (-1, 6, 2))

    return vertex_resize

def fix_vertex(contours: np.array)-> list:
    '''
    꼭지점 좌표들을 원하는 순서대로 정렬해주는 함수
    contours : 정렬되지 않은 6개의 꼭지점 좌표
    output -> contours : 정렬된 좌표
    예시)
    vertexes = utils.find_vertex(re_bg)
    object_vertexes =  utils.find_object_vertex(vertexes)
    object_vertexes = utils.fix_vertex(object_vertexes)
    '''
    # 최소 y좌표
    y_coors = np.min(contours, axis=0)[1]
    # print("y_coor", y_coors)
    # 좌상단 좌표가 index 0 번 -> 반시계 방향으로 좌표가 돌아간다.
    contours = contours.tolist()
    while y_coors != contours[-1][1]:
        temp = contours.pop(0)
        contours.append(temp)
    return contours

def find_object_vertex(vertexes:list, pts1: list) -> list:
    """
    꼭지점 좌표들 중 체커보드가 포함된 좌표들을 제거하는 함수
    vertexes : 좌표들의 리스트
    pts1 : 체커 포인트 최외곽 좌표 (4개)
    예시)
    vertexes = utils.find_vertex(re_bg)
    object_vertexes =  find_object_vertex(vertexes)
    """
    for vertex in vertexes:
        x_min, y_min = np.min(vertex, axis=0)
        x_max, y_max = np.max(vertex, axis=0)

        if not ((x_min < pts1[1][0] < x_max) and (y_min < pts1[1][1] < y_max)):
            vertexes = vertex
            break
    # print("vertex", vertex)
    return vertex

