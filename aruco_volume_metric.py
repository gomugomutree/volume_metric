import numpy as np
import glob, cv2
from matplotlib import pyplot as plt
import utils
from rembg import remove
import cv2.aruco as aruco
import pickle

class aruco_volumetric:
    
    def __init__(self, image_address: str, calibration_file_path: str, aruco_dict=aruco.DICT_6X6_1000, n_cluster=4, markerLength=1, aruco_real_size=10):

        self.img = cv2.imread(image_address)
        self.aruco_dict = aruco_dict # aruco.DICT_6X6_1000
        self.markerLength= markerLength 
        self.aruco_real_size = aruco_real_size # (cm)
        
        self.h = self.img.shape[0]
        self.w = self.img.shape[1]

        self.calibration_file_path = calibration_file_path
        self.camera_matrix = np.array
        self.dist = np.array
        self.tvecs = np.array
        
        self.re_bg_img = np.array # 
        self.refined_corners = np.array # 

        self.outer_points1 = np.array
        self.outer_points2 = np.array
        self.K = n_cluster
        self.object_detected_img = np.array
        self.vertexes = np.array
        self.object_vertexes = np.array
        self.transform_matrix = np.array

        self.width = float
        self.vertical = float
        self.height = 0.0

        self.height_pixel = np.array
    
    def set_calib_file(self):
        # if self.calibration_file_path.endswith('.npz'):
        #     with np.load(self.calibration_file_path) as X:
        #         self.camera_matrix, self.dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        #     print('The npz file is loaded')

        # elif self.calibration_file_path.endswith('.pckl'):
        with open(self.calibration_file_path, 'rb') as f:
            data = pickle.load(f)
            self.camera_matrix, self.dist, _, _ = data
        #     print('The pickle file is loaded')
            
        # else:
        #     print('Need to calibrate')


    # 배경 제거
    def remove_background(self):
        self.re_bg_img = remove(self.img)

    
    def find_ArucoMarkers(self): 
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        ARUCO_PARAMETERS = aruco.DetectorParameters_create()

        #ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000) original
        ARUCO_DICT = aruco.Dictionary_get(self.aruco_dict)
        corners, _, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        print(corners)

        self.refined_corners = corners

        self.rvecs, self.tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.camera_matrix, self.dist) # solvePnP                                       

    def find_aruco_corner(self, printer=False)-> np.array: #points: np.ndarray, size: tuple) -> np.ndarray:
        """
        points: cv2.cornerSubPix()에 의해 생성된 점들
        size: 체스판의 크기
        """
        corners = self.refined_corners
        
        self.outer_points1 =  np.array(corners).reshape(-1, 2)

        if printer:
            image = self.img.copy()
            self.image = cv2.circle(self.image, tuple(map(int,self.outer_points1[0])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            self.image = cv2.circle(self.image, tuple(map(int,self.outer_points1[1])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            self.image = cv2.circle(self.image, tuple(map(int,self.outer_points1[2])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            self.image = cv2.circle(self.image, tuple(map(int,self.outer_points1[3])), 3, (0, 0, 255), -1, cv2.LINE_AA )
            self.image = cv2.resize(self.image, (self.w //2, self.h //2))

            cv2.imshow("image", self.image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        #################
        # 급한데로 대충 맞춰서 outer_points1 정렬 
        # 자동 정렬 만들어야함

        self.outer_points1 = list(map(lambda x: x.tolist(), self.outer_points1))
        a, b, c, d = self.outer_points1
        self.outer_points1 = np.float32([a, d, b, c])

        # print("수정 후 : self.outer_points1:", self.outer_points1)

        ##############
   
    def find_object_by_k_mean(self, visualize_object=False):
        """
        배경제거된 이미지에서 물체의 그림자 등을 제거하는 이미지 전처리 과정 함수
        image : image matrix
        K : k mean cluster 개수
        visualize_object : 시각화 여부
        """

        img = cv2.cvtColor(self.re_bg_img, cv2.COLOR_BGR2RGB)
        twoDimage = img.reshape((-1,3))
        twoDimage = np.float32(twoDimage)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts=10

        _, label, center=cv2.kmeans(twoDimage, self.K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)

        res = center[label.flatten()]
        self.object_detected_img = res.reshape((img.shape))

        
        if visualize_object:
            result_img = cv2.resize(self.object_detected_img, (800, 800))
            cv2.imshow("img", result_img)
            cv2.waitKey(0)


    def find_vertex(self, printer=False):
        '''
        물체 꼭지점 6좌표 추출하는 함수
        iamge : 꼭지점을 찾을 이미지

        output -> vertex_resize : 꼭지점 좌표가 6개인 물체들의 좌표 리스트
        '''
        gray = cv2.cvtColor(self.object_detected_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)   # (1, 1)을 홀수값 쌍으로 바꿀수 있음 3,3 5,5 7,7.... 조절해가며 contours 상자를 맞춰감

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.erode(blurred, kernel, iterations =2)

        # Find contours
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours는 튜플로 묶인 3차원 array로 출력

        # vertex 출력값 그려보기
        vertex_list = list()
        for cnt in contours:
            for eps in np.arange(0.001, 0.2, 0.001):
                length = cv2.arcLength(cnt, True)
                epsilon = eps * length
                vertex = cv2.approxPolyDP(cnt, epsilon, True)
                if len(vertex) == 6 and length > 1000:      # vertex가 6 -> 꼭짓점의 갯수
                    
                    if printer:
                        image = self.img.copy()
                        cv2.drawContours(image,[vertex],0,(0,0,255),10)
                        image = cv2.resize(image, (self.w // 4, self.h // 4))
                        cv2.imshow(f"image", image)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

                    vertex_list.append(vertex)
                    break

        self.vertexes = np.reshape(vertex_list, (-1, 6, 2))
        if len(self.vertexes) == 0:
            print("object vertexes are not detected....")
            quit()      

    def find_object_vertex(self, printer=False):
        """
        꼭지점 좌표들 중 체커보드가 포함된 좌표들을 제거하는 함수
        vertexes : 좌표들의 리스트
        pts1 : 체커 포인트 최외곽 좌표 (4개)
        예시)
        vertexes = utils.find_vertex(re_bg)
        object_vertexes =  find_object_vertex(vertexes)
        """
        for vertex in self.vertexes:
            x_min, y_min = np.min(vertex, axis=0)
            x_max, y_max = np.max(vertex, axis=0)

            # 체커보드 좌표 안에 없는 꼭지점들을 찾아라
            if not ((x_min < self.outer_points1[1][0] < x_max) and (y_min < self.outer_points1[1][1] < y_max)):
                self.object_vertexes = vertex
                break
        if printer:
            print("물체 꼭지점 좌표 :", self.object_vertexes)

    def fix_vertex(self):
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
        y_coors = np.min(self.object_vertexes, axis=0)[1]
        # print("y_coor", y_coors)
        # 좌상단 좌표가 index 0 번 -> 반시계 방향으로 좌표가 돌아간다.
        contours = self.object_vertexes.tolist()
        while y_coors != contours[-1][1]:
            temp = contours.pop(0)
            contours.append(temp)
        self.object_vertexes = np.array(contours)


    
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
        self.transform_matrix = cv2.getPerspectiveTransform(self.outer_points1, self.outer_points2)


    def measure_width_vertical(self, printer=False):
        """
        checker_points : checker board 기준 4개 좌표
        object_points : 물체의 외곽선 좌표 
        aruco_real_dist : 체크 1칸의 실제 거리 (cm)
        mat : 변환 행렬 (3 X 3) - cv2.getPerspectiveTransform(pts1, pts2)한 결과값
        printer : 가로, 세로 길이 출력문 실행 여부 - bool
        result : [가로 길이, 세로 길이] 

        """
        re_point = list()
        checker_points = self.outer_points1
        checker_points = checker_points.tolist()
        # 체커보드가 정방향으로 투시되었을때 각 좌표들을 다시 구해준다.
        for point in checker_points:
            # print("point",type(point))
            re_point.append(utils.transform_coordinate(self.transform_matrix, point))

        re_object_points = list()
        re_checker_points = self.object_vertexes.tolist()

        for point in re_checker_points:
            re_object_points.append(utils.transform_coordinate(self.transform_matrix, point))

        # pt2[0]의 x축과 pt2[2]의 x축의 필셀 거리 // 코너 사이즈 - 1 (칸) = 1칸당 떨어진 픽셀거리
        one_checker_per_pix_dis = abs(re_point[0][0] - re_point[2][0])

        # 픽셀당 실제 거리 - aruco_real_dist(cm) / 1칸당 떨어진 픽셀 거리
        pix_per_real_dist = self.aruco_real_dist / one_checker_per_pix_dis

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


    def measure_height(self, printer=False):

        pts1 = self.outer_points1.tolist()

        ar_start = utils.transform_coordinate(self.transform_matrix, pts1[0])
        ar_second = utils.transform_coordinate(self.transform_matrix, pts1[2])

        vertexes_list = self.object_vertexes[1].tolist()
  
        self.center = np.mean(self.outer_points2, axis=0).tolist()

        ar_object_standard_z = utils.transform_coordinate(self.transform_matrix, vertexes_list)
        # print(ar_object_standard_z)

        # 두 점을 1으로 나눈 거리를 1칸 기준 (ckecker 사이즈에서 1 빼면 칸수)
        standard_ar_dist = abs(ar_start[0] - ar_second[0])
        
        # 실제세계의 기준 좌표를 기준으로 물체의 z축을 구할 바닥 좌표의 실제세계의 좌표를 구한다
        # x, y, z 값을 갖는다
        ar_object_real_coor = [
            (ar_object_standard_z[0] - self.center[0]) / standard_ar_dist,
            -1 *(ar_object_standard_z[1] - self.center[1]) / standard_ar_dist,
            0,
        ]
        print("ar_object_real_coor :", ar_object_real_coor)
 
        self.rvecs = self.rvecs.reshape(3, 1) # (1, 1, 3) -> (3, 1)
        self.tvecs = self.tvecs.reshape(3, 1) # (1, 1, 3) -> (3, 1)

        height_pixel = utils.pixel_coordinates(self.camera_matrix, self.rvecs, self.tvecs, ar_object_real_coor)
        
        for i in np.arange(0, 20, 0.01):
            if (height_pixel[1] - self.object_vertexes[0][1]) < 0:
                break

            height_pixel = utils.pixel_coordinates(
                self.camera_matrix, self.rvecs, self.tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], i)
            )
            self.height = i
            if printer:
                self.img = cv2.circle(self.img, tuple(list(map(int, height_pixel[:2]))), 3, (255, 0, 255), -1, cv2.LINE_AA)
        
        
    def draw_image(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 가로, 세로, 높이 출력
        print("가로길이 :",self.width)
        print("세로길이 :",self.vertical)
        print("높이길이 :",self.height * self.aruco_real_size)
        print(f"{self.width: .2f} x {self.vertical: .2f} x {(self.height * self.aruco_real_size): .2f}")

        # 가로세로 그리기
        cv2.putText(self.img, f"width:{self.width: .2f} length: {self.vertical: .2f} height: {self.height * self.aruco_real_size: .2f}", (self.w//5, self.object_vertexes[2][1]+100), font, 3, (255, 0, 0), 10)

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


def main(image, npz, real_dist):
    a = aruco_volumetric(image, npz, real_dist)
    a.set_calib_file()
    a.find_ArucoMarkers()
    a.find_aruco_corner()
    a.remove_background()

    a.find_object_by_k_mean()
    a.find_vertex()
    a.find_object_vertex(printer=True)
    a.fix_vertex()
    a.trans_checker_stand_coor()
    a.set_transform_matrix()
    a.measure_width_vertical()
    a.measure_height(printer=True)
    a.draw_image()

# for i in range(24, 25):
main(f"./aruco_image/aruco_img10.jpg", "aruco_calibration.pckl", 1)


