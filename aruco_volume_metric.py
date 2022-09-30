import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
import utils
import cv2.aruco as aruco
# from rembg import remove
import pickle


class volumetric:
    
    def __init__(self, image_address: str, npz_file: str, aruco_dict=aruco.DICT_6X6_1000, n_cluster=4):

        self.img = cv2.imread(image_address)

        self.h = self.img.shape[0]
        self.w = self.img.shape[1]

        self.npz_file = npz_file
        self.camera_matrix = np.array
        self.dist = np.array
        self.tvecs = np.array
        self.rvecs = np.array
        self.re_bg_img = np.array # 
        self.refined_corners = np.array # 
        self.checker_sizes = tuple()
        self.outer_points1 = np.array
        self.outer_points2 = np.array
        self.K = n_cluster
        self.object_detected_img = np.array
        self.vertexes = np.array
        self.object_vertexes = np.array
        self.transform_matrix = np.array
        self.aruco_dict = aruco_dict

        self.width = float
        self.vertical = float
        self.height = 0.0

        self.height_pixel = np.array

    def set_image(self, image_address: str):
        self.img = cv2.imread(image_address)


    def set_npz(self):
        # with np.load(self.npz_file) as X:
        #     self.camera_matrix, self.dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        with open('calibration.pckl', 'rb') as f:
            data = pickle.load(f)
            self.camera_matrix, self.dist = data


    def find_ArucoMarkers(self): #img, markerSize = 6, totalMarkers=250, draw=True):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
        # arucoDict = aruco.Dictionary_get(key)
        # arucoParam = aruco.DetectorParameters_create()
        # bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
        # print(ids)
        ARUCO_PARAMETERS = aruco.DetectorParameters_create()

        #ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000) original
        ARUCO_DICT = aruco.Dictionary_get(self.aruco_dict)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        # Create grid board object we're using in our stream
        board = aruco.GridBoard_create(
                markersX=2,
                markersY=2,
                markerLength=0.09,
                markerSeparation=0.01,
                dictionary=ARUCO_DICT)


        self.refined_corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers( # cornerSubPix
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = self.camera_matrix,
                distCoeffs = self.dist)   

        self.rvecs, self.tvecs, _objPoints = aruco.estimatePoseSingleMarkers(self.refined_corners, 0.09, self.camera_matrix, self.dist) # solvePnP                                       
        print("self.refined_corners :", self.refined_corners)

        # # output
        # self.refined_corners : (array([[[ 534.,  703.],
        # [ 868.,  608.],
        # [1016.,  816.],
        # [ 638.,  936.]]], dtype=float32),)

        
   
   ###########################
 
    def find_checker_outer_points(self, printer=False)-> np.array: #points: np.ndarray, size: tuple) -> np.ndarray:
        """
        points: cv2.cornerSubPix()에 의해 생성된 점들
        size: 체스판의 크기
        """
        temps = self.refined_corners
        print(type(temps)) # 튜플
        
        self.outer_points1 =  np.array(temps).reshape(-1, 2)

        if printer:
            image = self.img.copy()
            image = cv2.circle(image, tuple(map(int,self.outer_points1[0])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            image = cv2.circle(image, tuple(map(int,self.outer_points1[1])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            image = cv2.circle(image, tuple(map(int,self.outer_points1[2])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            image = cv2.circle(image, tuple(map(int,self.outer_points1[3])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            image = cv2.resize(image, (self.w // 2, self.h // 2))
            
            cv2.imshow(f"image", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        #################
        # 급한데로 대충 맞춰서 outer_points1 정렬 
        # 자동 정렬 만들어야함

        self.outer_points1 = list(map(lambda x: x.tolist(), self.outer_points1))
        a, b, c, d = self.outer_points1
        self.outer_points1 = np.float32([a, d, b, c])

        print("수정 후 : self.outer_points1:", self.outer_points1)

        ##############

    
    def trans_checker_stand_coor(self):# point: list, stand_corr: tuple, checker_size: tuple) -> list:
        """
        ** 수정 필요 **
        이미지상의 4개의 좌표를 일정한 간격으로 펴서 4개의 좌표로 만들어주는 함수
        point : ex) np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
        stand_coor : 새로 만드는 좌표의 좌측 상단의 기준 좌표
        result : point와 같은 형식 
        """
        # x, y 비율과 똑같이 ar 이미지에 투시한다.
        # 첫번째 좌표를 기준으로 오른쪽에서 x, 아래쪽 좌표에서 y 간격(비율)을 구해준다.
        # 1칸당 거리 구하기
        one_step = abs(self.outer_points1[0][0] - self.outer_points1[2][0])


        w, h = (self.w, self.h * 2)
        self.outer_points2 = np.float32(
            [[w, h], 
            [w, h + one_step ], 
            [w + one_step , h], 
            [w + one_step, h + one_step ],]
        )

    # 투시 행렬 구하기
    def set_transform_matrix(self):
        print(self.outer_points1)
        print(self.outer_points2)
        print(self.outer_points1.shape, self.outer_points2.shape)
        print(type(self.outer_points1))
        print(type(self.outer_points2))
        self.transform_matrix = cv2.getPerspectiveTransform(self.outer_points1, self.outer_points2)

    def measure_width_vertical(self, printer=False):
    #     checker_points: list,
    #     object_points: list,
    #     check_real_dist: int,
    #     mat: np.array,
    #     checker_size: tuple,
    #     printer=True,
    # ) -> list:
        """
        checker_points : checker board 기준 4개 좌표
        object_points : 물체의 외곽선 좌표 
        check_real_dist : 체크 1칸의 실제 거리 (cm)
        mat : 변환 행렬 (3 X 3) - cv2.getPerspectiveTransform(pts1, pts2)한 결과값
        printer : 가로, 세로 길이 출력문 실행 여부 - bool
        result : [가로 길이, 세로 길이] 

        """
        re_point = list()
        checker_points = self.outer_points1
        checker_points = checker_points.tolist()
        # 체커보드가 정방향으로 투시되었을때 각 좌표들을 다시 구해준다.
        for point in checker_points:
            print("point",type(point))
            re_point.append(utils.transform_coordinate(self.transform_matrix, point))

        re_object_points = list()
        re_checker_points = self.object_vertexes.tolist()

        for point in re_checker_points:
            re_object_points.append(utils.transform_coordinate(self.transform_matrix, point))

        # pt2[0]의 x축과 pt2[2]의 x축의 필셀 거리 // 코너 사이즈 - 1 (칸) = 1칸당 떨어진 픽셀거리
        one_checker_per_pix_dis = abs(re_point[0][0] - re_point[2][0]) / (
            self.checker_sizes[0] - 1
        )

        # 픽셀당 실제 거리 - check_real_dist(cm) / 1칸당 떨어진 픽셀 거리
        pix_per_real_dist = self.check_real_dist / one_checker_per_pix_dis

        # 두 점 사이의 픽셀거리 * 1픽셀당 실제 거리 = 두 점의 실제 거리
        self.width = (
            utils.euclidean_distance(re_object_points[1], re_object_points[2]) * pix_per_real_dist
        )
        self.vertical = (
            utils.euclidean_distance(re_object_points[2], re_object_points[3]) * pix_per_real_dist
        )

        if printer:
            print("1칸당 픽셀거리 :", one_checker_per_pix_dis)
            print("픽셀당 실제 거리 :", pix_per_real_dist)
            print("가로길이 :", self.width)
            print("세로길이 :", self.vertical)


    def temp_draw(self):

        # pts1 = self.outer_points1.tolist()
        # ar_start = utils.transform_coordinate(self.transform_matrix, pts1[0])
        # ar_second = utils.transform_coordinate(self.transform_matrix, pts1[2])
        
        # vertexes_list = self.object_vertexes[1].tolist()
        # ar_object_standard_z = utils.transform_coordinate(self.transform_matrix, vertexes_list)

        # # 두 점을 1으로 나눈 거리를 1칸 기준 (ckecker 사이즈에서 1 빼면 칸수)
        # standard_ar_dist = abs(ar_start[0] - ar_second[0]) / (self.checker_sizes[0] - 1)  
        
        # # 실제세계의 기준 좌표를 기준으로 물체의 z축을 구할 바닥 좌표의 실제세계의 좌표를 구한다
        # # x, y, z 값을 갖는다
        # #
        # ar_object_real_coor = [
        #     (ar_object_standard_z[0] - ar_start[0]) / standard_ar_dist,
        #     (ar_object_standard_z[1] - ar_start[1]) / standard_ar_dist,
        #     0,
        # ]
        ar_object_real_coor = [0, 0, 0]
        # pixel_coordinates 
        print("rvecs", self.rvecs, self.rvecs.shape) # (1, 1, 3)
        print("tvecs", self.tvecs, self.tvecs.shape) # (1, 1, 3)
        
        self.rvecs = self.rvecs.reshape(3, 1) # (1, 1, 3) -> (3, 1)
        self.tvecs = self.tvecs.reshape(3, 1) # (1, 1, 3) -> (3, 1)

        
        height_pixel = utils.pixel_coordinates(self.camera_matrix, self.rvecs, self.tvecs, ar_object_real_coor)

        # x축
        for i in np.arange(0, 0.1, 0.01):
            # if (height_pixel[1] - self.object_vertexes[0][1]) < 0:
            #     break
            height_pixel = utils.pixel_coordinates(
                self.camera_matrix, self.rvecs, self.tvecs, (ar_object_real_coor[0]+i, ar_object_real_coor[1], 0)
            )
            self.height = i
            self.img = cv2.circle(self.img, tuple(list(map(int, height_pixel[:2]))), 5, (0, 255, 0), -1, cv2.LINE_AA)
        # y축
        for i in np.arange(0, 0.1, 0.01):
            # if (height_pixel[1] - self.object_vertexes[0][1]) < 0:
            #     break
            height_pixel = utils.pixel_coordinates(
                self.camera_matrix, self.rvecs, self.tvecs, (ar_object_real_coor[0], ar_object_real_coor[1]+i, 0)
            )
            self.height = i
            self.img = cv2.circle(self.img, tuple(list(map(int, height_pixel[:2]))), 5, (255, 0, 0), -1, cv2.LINE_AA)
        # z축
        for i in np.arange(0, 0.1, 0.01):
            # if (height_pixel[1] - self.object_vertexes[0][1]) < 0:
            #     break
            height_pixel = utils.pixel_coordinates(
                self.camera_matrix, self.rvecs, self.tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], -i)
            )
            self.height = i
            self.img = cv2.circle(self.img, tuple(list(map(int, height_pixel[:2]))), 5, (0, 0, 255), -1, cv2.LINE_AA)

        
        self.img = cv2.resize(self.img, (self.w // 4, self.h // 4))

        cv2.imshow("img", self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()



    def draw_image(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 가로, 세로, 높이 출력
        print("가로길이 :",self.width)
        print("세로길이 :",self.vertical)
        print("높이길이 :",self.height * self.check_real_dist)
        print(f"{self.width: .2f} x {self.vertical: .2f} x {(self.height * self.check_real_dist): .2f}")

        # 가로세로 그리기
        cv2.putText(self.img, f"width:{self.width: .2f} vertical: {self.vertical: .2f} height: {self.height*4}", (self.w//5, self.object_vertexes[2][1]+100), font, 3, (255, 0, 0), 10)

        cv2.line(self.img,(self.object_vertexes[1]), (self.object_vertexes[2]), (0, 255, 0), 5, cv2.LINE_AA)
        cv2.line(self.img,(self.object_vertexes[2]), (self.object_vertexes[3]), (255, 0, 0), 5, cv2.LINE_AA)
        # cv2.line(self.img,(self.object_vertexes[1]), (self.height_pixel), (255, 0, 0), 5, cv2.LINE_AA)

        self.img = cv2.resize(self.img, (self.w // 4, self.h // 4))

        cv2.imshow("img", self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def show_image(self, image: np.array):
        image = cv2.resize(image, (self.w // 4, self.h // 4))
        cv2.imshow(f"image", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main(image, npz):
    a = volumetric(image, npz)
    a.set_npz()
    # a.show_image(a.re_bg_img)
    a.find_ArucoMarkers()
    # a.search_aruco()
    a.find_checker_outer_points()

    a.trans_checker_stand_coor()
    a.set_transform_matrix()
    # a.draw_image()
    a.temp_draw()

# for i in range(24, 25):
#     main(f"./image/img{i}.jpg", "calibration3.npz", 4)
for i in range(1, 4):
    main(f"./aruco_image/aruco{i}.jpg", "calibration.pckl")

