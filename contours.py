# Checkerboard 부분 잘라내기
import cv2
import numpy as np
import random


image = cv2.imread('20220922_111748.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3,3), 0)

edged = cv2.Canny(blurred, 50, 180)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

dilate = cv2.dilate(edged, kernel, iterations=3)

contours, hier = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()


mask = np.zeros(image.shape).astype(image.dtype)

color = [255, 255, 255]

# 경계선 내부 255로 채우기

test = cv2.fillPoly(mask, contours, color)

# 4차원 contours를 2차원으로 축소
# contours_reshape = np.reshape(contours, (-1, 2))
contours_reshape = np.squeeze(contours[0])
contours_reshape.shape

# 각 축의 최대, 최소를 리스트로 정리
x_coor = []
y_coor = []
for i in range(len(contours_reshape)):
    x_coor.append(contours_reshape[i][0])
    y_coor.append(contours_reshape[i][1])


crop_img = image[0:min(y_coor), 0:int(image.shape[0])] # Crop from x, y, w, h -> 100, 200, 300, 400




# =============================================================================================
# rembg를 사용한 물체 배경제거


from rembg import remove
import cv2

# input_path = 'input.png'
# output_path = 'output.png'

# input = cv2.imread(input_path)
output = remove(crop_img)

# cv2.imwrite('rembg_image.jpg', output) # 새로운 이미지로 저장한 뒤 사용 가능

# output = cv2.resize(output, dsize = (500, 250), interpolation = cv2.INTER_CUBIC)
# cv2.imshow("test", output)
# cv2.waitKey(0)






# =============================================================================================
#배경제거된 물체 이미지 외곽선 추출 및 꼭짓점 좌표 추출

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (1, 1), 0)   # (1, 1)을 홀수값 쌍으로 바꿀수 있음 3,3 5,5 7,7.... 조절해가며 contours 상자를 맞춰감
edged = cv2.Canny(blurred, 6, 7)              # 6, 7을 바꿀수 있음 조절해가며 contours 상자를 맞춰감

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)    # 아래의 dilate 또는 closed를 사용
dilate = cv2.dilate(edged, kernel, iterations = 1)


# Find contours
contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours는 튜플로 묶인 3차원 array로 출력

# Make a mask image
mask = np.zeros(image.shape).astype(image.dtype)

color = [255,255,255]
filled_image = cv2.fillPoly(mask, contours, color)

contours_resize = np.squeeze(contours)


contours_coor_list = [val for val in contours_resize]  # 2차원 리스트로 출력
# e.g.) [[x1, y1], [x2, y2], ...]
# print(type(contours_coor_list))
# print(contours_coor_list[0][0][0][0])
# x좌표와 y좌표를 분할

x_coors = [coor[0][0][0] for coor in contours_coor_list]  # x 좌표만 추출
y_coors = [coor[0][0][1] for coor in contours_coor_list]  # y 좌표만 추출

# print(max(x_coors), min(x_coors), max(y_coors), min(y_coors))  # x 및 y 좌표 집단에서 최대, 최소 추출
# print(min(x_coors))

# 물체를 꽉 채우는 상자 생성
cv2.rectangle(
    filled_image,
    (min(x_coors), min(y_coors)),
    (max(x_coors), max(y_coors)),
    (0, 255, 0),
    10)

# =============================================================================================




# 꼭짓점 구하기 1
# approx = cv2.approxPolyDP(contours[1], cv2.arcLength(contours[1], True) * 0.02, True)



# 외곽선 따라 그리기
for cnt in contours:
    cv2.drawContours(filled_image, [cnt], 0, (255, 0, 0), 10)


# approx 출력값 그려보기
# for cnt in contours:
#     epsilon = 0.001 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     if len(approx) == 6:      # approx가 6 -> 꼭짓점의 갯수
#         cv2.drawContours(filled_image,[approx],0,(0,0,255),20)




#꼭짓점 구하기   2
approx_fin = cv2.approxPolyDP(contours[1], cv2.arcLength(contours[1], True) * 0.00001, True) # contours[?] ?에 따라 다른 값들이 나옴.
# approx_resize = np.reshape(approx, (-1, 2))
# print(approx)



# # 선분 길이 구하기 (가로, 세로, 높이 in pixels)
# dist = []
# for i in range(len(approx_resize)):
#     dist.append(int(np.linalg.norm(approx_resize[i - 1] - approx_resize[i])))
# # print(dist)

# print(np.array(contours).shape)
# print(contours)

print('꼭짓점:' , approx_fin)
# print(approx)
# print('선분:', dist)

filled_image = cv2.resize(filled_image, dsize = (500, 500), interpolation = cv2.INTER_CUBIC)
cv2.imshow("test", filled_image)
cv2.waitKey(0)

