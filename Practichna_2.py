import cv2
import numpy as np

img = cv2.imread('images/prac2_img1.jpg')
img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
img_copy = img.copy()

img = cv2.GaussianBlur(img, (5, 5), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_green = np.array([41, 29, 0])
upper_green = np.array([104, 255, 137])

lower_red = np.array([141, 41, 132])
upper_red = np.array([179, 225, 255])

lower_blue = np.array([70, 35, 0])
upper_blue = np.array([126, 225, 255])

lower_yellow = np.array([17, 33, 0])
upper_yellow = np.array([26, 225, 255])



mask_red = cv2.inRange(img, lower_red, upper_red)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)
mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)


contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_y, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)





for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True) #ОБЧИСЛЮЄТЬСЯ ПЕРИМЕТР КОНТУРУ
        M = cv2.moments(cnt)                #моменти конкуру

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        aspect_ratio = round(w/h,2)
        compactness = round((4* np.pi * area)/(perimeter ** 2),2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) #ккість вершин у фігурі
        if len(approx) == 4:
            shape = "square"
        elif len(approx) == 3:
            shape = "triangle"
        elif len(approx) >5 and len(approx)<=7:
            shape = "else"
        else:
            shape = "oval"





        cv2.putText(img_copy, f'S: {int(area)}, P:{int(perimeter)}', (x+w, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_copy, f'x: {x}, y:{y}', (x + w, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_copy, (cx,cy),  4, (0, 0, 255), -1)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x+w, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_copy, f'shape:{shape}', (x+w, y +35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)



for cnt1 in contours_g:
    area = cv2.contourArea(cnt1)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt1)
        color = "green"
        cv2.putText(img_copy, f'color:{color}', (x + w, y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
for cnt2 in contours_b:
    area = cv2.contourArea(cnt2)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt2)
        color = "blue"
        cv2.putText(img_copy, f'color:{color}', (x + w, y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
for cnt3 in contours_r:
    area = cv2.contourArea(cnt3)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt3)
        color = "red"
        cv2.putText(img_copy, f'color:{color}', (x + w, y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
for cnt4 in contours_y:
    area = cv2.contourArea(cnt4)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt4)
        color = "yellow"
        cv2.putText(img_copy, f'color:{color}', (x + w, y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("result.jpg", img_copy)
cv2.imshow('img', img)
cv2.imshow('mask', img_copy)



cv2.waitKey(0)

cv2.destroyAllWindows()