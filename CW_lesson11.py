import pandas as pd # для файлів csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')  #робота з csv
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']] #обираэмо елементи для навчання
y = df['label_enc']

#створення моделі
model = keras.Sequential([layers.Dense(8, activation = 'relu', input_shape = (3,)),
                          layers.Dense(8, activation = 'relu'),
                          layers.Dense(8, activation = 'softmax')])

#навчення моделі, створення графіку
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X, y, epochs = 200, verbose = 0)
plt.plot(history.history['loss'], label = 'Втрата (Loss)')
plt.plot(history.history['accuracy'], label = 'Точність (Accuracy)')
plt.xlabel("Епоха")
plt.ylabel('Значення')
plt.title('Процес навчання')
plt.legend()
plt.show()

#тестування
test = np.array([18, 16, 0])

pred = model.predict(test)
print(f'Імовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')



