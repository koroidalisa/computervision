import cv2
import numpy as np

img = np.zeros((400, 600, 3), np.uint8)
img[:] = 232, 242, 143


cv2.putText(img, "Alisa Koroid", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
cv2.putText(img, "Computer Vision Student", (200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (79, 78, 78), 2)
cv2.putText(img, "Email: koroidalisa@gmail.com", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (105, 25, 1), 1)
cv2.putText(img, "Phone: +380680738655", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (105, 25, 1), 1)
cv2.putText(img, "24/08/2010", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (105, 25, 1), 1)
cv2.putText(img, "OpenCV Business Card", (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

me = cv2.imread('image/image_practichna.jpg')
print(me.shape)
me = cv2.resize(me, (me.shape[1]//9, me.shape[0]//9))
x, y = 30, 30
h, w = me.shape[:2]
img[y:y+h, x:x+w] = me

qr = cv2.imread('image/qr.png')
print(qr.shape)
qr = cv2.resize(qr, (qr.shape[1]//4, qr.shape[0]//4))
x1, y1 = 475, 210
h1, w1 = qr.shape[:2]

h_img, w_img = img.shape[:2]
if y1 + h1 > h_img:
    h1 = h_img - y1
if x1 + w1 > w_img:
    w1 = w_img - x1

img[y1:y1+h1, x1:x1+w1] = qr[0:h1, 0:w1]



cv2.rectangle(img, (10, 10), (590, 390), (155, 90, 30), 2)


cv2.imshow("image", img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()