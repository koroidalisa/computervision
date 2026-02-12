import cv2
import os
import csv
import yt_dlp
from ultralytics import YOLO

# Лінія СТАРТУ (Синя) - ближче до верху/середини (залежить від ракурсу)
LINE_START = [(200, 650), (1250, 800)]
# Лінія ФІНІШУ (Червона) - ближче до низу
LINE_END = [(1110, 365), (1600, 430)]


DISTANCE_METERS = 20
FRAME_SKIP = 2

# ПАПКИ
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, 'car_statistics.csv')
YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = "yolov8n.pt"


# ФУНКЦІЇ
def get_stream_url(url):
    ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f"Помилка стріму: {e}")
        return None


# Перевірка перетину відрізків
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# ЗАПУСК
model = YOLO(MODEL_PATH)
stream_url = get_stream_url(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Не вдалося відкрити відео.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Ініціалізація CSV
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'ID', 'Speed_kmh', 'Time_sec'])

# Змінні стану
tracker_data = {}  # Зберігає {id: {'frame': N, 'line': 'start'/'end'}}
previous_positions = {}  # Зберігає {id: (x, y)} попереднього кадру
car_speeds = {}  # Зберігає {id: speed_int}
unique_ids = set()
frame_count = 0

print("Обробка")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Трекінг (2=car, 3=motorcycle, 5=bus, 7=truck)
    results = model.track(frame, conf=0.4, persist=True, verbose=False, classes=[2, 3, 5, 7])

    # лінії
    cv2.line(frame, LINE_START[0], LINE_START[1], (255, 0, 0), 2)  # Blue
    cv2.line(frame, LINE_END[0], LINE_END[1], (255, 0, 0), 2)  # Red

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        unique_ids.update(ids)

        for box, tid, cls_id in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            curr_pos = (cx, cy)
            class_name = model.names[cls_id]

            if tid in previous_positions:
                prev_pos = previous_positions[tid]

                # ЛОГІКА ПЕРЕТИНУ

                # 1. Перетин СИНЬОЇ лінії
                if intersect(prev_pos, curr_pos, LINE_START[0], LINE_START[1]):
                    # Якщо машина їхала від Червоної до Синьої (завершує шлях)
                    if tid in tracker_data and tracker_data[tid]['line'] == 'end':
                        start_frame = tracker_data[tid]['frame']
                        end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        frames_diff = abs(end_frame - start_frame)

                        if frames_diff > 1:  # Захист від миттєвого глюку
                            time_sec = frames_diff / fps
                            speed = (DISTANCE_METERS / time_sec) * 3.6
                            car_speeds[tid] = int(speed)

                            # Запис у CSV
                            with open(CSV_PATH, mode='a', newline='') as f:
                                csv.writer(f).writerow([class_name, tid, int(speed), round(time_sec, 2)])

                            del tracker_data[tid]  # Очистка пам'яті для цього авто

                    # Якщо машина тільки заїхала на ділянку (починає шлях Blue->Red)
                    elif tid not in tracker_data:
                        tracker_data[tid] = {'frame': cap.get(cv2.CAP_PROP_POS_FRAMES), 'line': 'start'}

                # 2. Перетин ЧЕРВОНОЇ лінії (END)
                if intersect(prev_pos, curr_pos, LINE_END[0], LINE_END[1]):
                    # Якщо машина їхала від Синьої до Червоної (завершує шлях)
                    if tid in tracker_data and tracker_data[tid]['line'] == 'start':
                        start_frame = tracker_data[tid]['frame']
                        end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        frames_diff = abs(end_frame - start_frame)

                        if frames_diff > 1:
                            time_sec = frames_diff / fps
                            speed = (DISTANCE_METERS / time_sec) * 3.6
                            car_speeds[tid] = int(speed)

                            with open(CSV_PATH, mode='a', newline='') as f:
                                csv.writer(f).writerow([class_name, tid, int(speed), round(time_sec, 2)])

                            del tracker_data[tid]

                    # Якщо машина тільки заїхала (починає шлях Red->Blue)
                    elif tid not in tracker_data:
                        tracker_data[tid] = {'frame': cap.get(cv2.CAP_PROP_POS_FRAMES), 'line': 'end'}

            # Оновлюємо позицію
            previous_positions[tid] = curr_pos

            # --- МАЛЮЄМО РАМКИ ---
            color = (0, 255, 0)
            label = f"ID:{tid}"

            # Якщо швидкість відома, показуємо її
            if tid in car_speeds:
                color = (0, 0, 255)  # Червона рамка для порахованих
                label += f" {car_speeds[tid]} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- СТАТИСТИКА (ВИПРАВЛЕНО) ---
    # Розрахунок середньої швидкості
    if len(car_speeds) > 0:
        avg_speed = sum(car_speeds.values()) / len(car_speeds)
        avg_speed_text = f"Avg Speed: {avg_speed:.1f} km/h"
    else:
        avg_speed_text = "Avg Speed: 0.0 km/h"

    # Малюємо плашку
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Cars: {len(unique_ids)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Тут тепер показуємо середню швидкість
    cv2.putText(frame, avg_speed_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Traffic Speed Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()