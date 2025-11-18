import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train', image_size=(128, 128), batch_size=30, label_mode='categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test', image_size=(128, 128), batch_size=30, label_mode='categorical')
#нормалізація зображень
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
#побудова моделі
model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size = (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#фільтри - прості ознаки лінії і контури

model.add(layers.Conv2D(filters=64, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size = (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=15, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)

class_name = ['cars', 'cats', 'dogs']
img = image.load_img('images/', target_size=(128, 128))

img_array = image.img_to_array(img)

#нормалізуємо зображення
img_array = img_array/255.0
img_array = np.expand_dims(img_array, axis=0)

#прогноз
pred = model.predict(img_array)

pred_index = np.argmax(pred[0])
print(f'імовірність по класам: {pred[0]}')

print(f'модель визначила: {class_name[pred_index]}')