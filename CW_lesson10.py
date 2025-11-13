import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == 'circle':
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == 'square':
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == 'triangle':
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

#2 Формуємо набір даних
X = []   # список ознак
y = []   # список міток

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}
shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)

            y.append(f"{color_name}_{shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Точність моделі:", round(accuracy * 100, 2), "%")


test_img = generate_image((0, 0, 255), 'square')
mean_color = cv2.mean(test_img)[:3]
prediction = model.predict([mean_color])
print(f'Передбачення: {prediction[0]}')

cv2.imshow("test", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()