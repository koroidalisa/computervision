import cv2
from collections import Counter

# Список цільових файлів для обробки
TARGET_FILES = [
    "shark.jpg",
    "cat.jpg",
    "crab.jpg",
    "saxop.jpg",
    "hen.jpg"
]

# 1) Завантажуємо попередньо навчену модель MobileNet
net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

# 2) Читаємо список назв класів
classes = []
# Шлях до файлу класів прописаний напряму
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

classification_results = Counter()

# Шлях до папки зображень
IMAGE_FOLDER = 'image/MobileNet/'
print(f"Обробка зображень з папки: {IMAGE_FOLDER}")

for filename in TARGET_FILES:
    # 3) Формування шляху до зображення конкатенацією (запобігає помилкам зі слешами)
    full_path = IMAGE_FOLDER + filename
    print(f"\n---> Обробка файлу: {filename}")

    image = cv2.imread(full_path)

    if image is None:
        print(f"Помилка: Не вдалося завантажити зображення {filename}. Перевірте шлях: {full_path}")
        continue

    # 4) Готуємо зображення для мережі: створюємо blob (тензор)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    # 5) Кладемо підготовлені дані в мережу і запускаємо "forward pass"
    net.setInput(blob)
    preds = net.forward()

    # 6) Знаходимо індекс класу з найбільшою ймовірністю
    idx = preds[0].argmax()

    # 7) Дістаємо назву класу і впевненість (у відсотках)
    label = classes[idx] if idx < len(classes) else "unknown"
    conf = float(preds[0][idx]) * 100
    conf_rounded = round(conf, 2)

    # 8) Вивід результатів
    print("Клас:", label)
    print("Ймовірність:", conf_rounded, "%")

    classification_results[label] += 1

    # 9) Підпис на зображенні
    text = label + ": " + str(int(conf)) + "%"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 9) Підпис на зображенні
    text = label + ": " + str(int(conf)) + "%"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # !!! ДОДАТИ ЦІ ДВА РЯДКИ ДЛЯ ПОКАЗУ !!!
    cv2.imshow(f"Результат: {filename}", image)
    cv2.waitKey(0)  # Зачекати, доки користувач натисне клавішу, щоб перейти до наступного фото

cv2.destroyAllWindows()


print("ЗВЕДЕНА ТАБЛИЦЯ КЛАСИФІКАЦІЇ")


COL_WIDTH_CLASS = 40
COL_WIDTH_COUNT = 10

header_class = "Зустрінутий клас"
header_count = "Кількість"
print(f"{header_class:<{COL_WIDTH_CLASS}} | {header_count:<{COL_WIDTH_COUNT}}")
print("-" * (COL_WIDTH_CLASS + COL_WIDTH_COUNT + 3))

for cls, count in classification_results.items():
    print(f"{cls:<{COL_WIDTH_CLASS}} | {count:<{COL_WIDTH_COUNT}}")

print("=" * 50)