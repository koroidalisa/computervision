import cv2
import numpy as np
image = cv2.imread("images/imagehw2902.jpg")
image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
print(image.shape)
cv2.rectangle(image, (180, 170), (410, 390), 3)
cv2.putText(image, "Alisa Koroid", (192, 415), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))


cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()