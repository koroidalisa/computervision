import cv2
import numpy as np

# image = cv2.imread("images/image11.jpg")
# #image = cv2.resize(image, (800, 400))
# image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
# #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# #image = cv2.flip(image, 1)
# #image = cv2.GaussianBlur(image, (9, 9), 0) #лише непарні числа в передостанній аргумент(9, 9)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image.shape)
# image = cv2.Canny(image, 100, 100)
# #1 Жирність контура
# #image = cv2.dilate(image, None, iterations=1)
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# image = cv2.erode(image, kernel, iterations=1)
#
#
#
#
# cv2.imshow("kit", image)
# #cv2.imshow("kitik", image[0:100, 0:200]) якщо треба обрізати фрагмент

#video = cv2.VideoCapture("video/vid1.mp4")
video = cv2.VideoCapture(0)
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("videozubr", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#cv2.waitKey(0)
cv2.destroyAllWindows()
