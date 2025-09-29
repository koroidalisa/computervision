import cv2
import numpy as np

# image = cv2.imread("image/imagehw2902.jpg")
# print(image.shape)
# image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
#
#
# image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# image = cv2.Canny(image, 150, 150)
#
# cv2.imshow("me", image)

image2 = cv2.imread("image/imagehw2902_2.jpg")
print(image2.shape)
image2 = cv2.resize(image2, (image2.shape[1]//2, image2.shape[0]//2))


image2= cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

image2 = cv2.Canny(image2, 300, 300)

cv2.imshow("me", image2)



cv2.waitKey(0)
cv2.destroyAllWindows()