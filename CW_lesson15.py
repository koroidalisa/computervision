import os
import cv2
import time
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)


USE_WEBCAM = True

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'name')
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO('yolov8n.pt')

CONF_TRESHOLD = 0.4

RESIZE_WIDTH = 960 #None

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        break


    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]

        scale = RESIZE_WIDTH/w

        new_w = int(scale * w)
        new_h = int(scale*h)

        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf = CONF_TRESHOLD, verbose = False)
    people_count = 0
    psevdo_id = 0

    PERSON_CLASS_ID = 0

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_CLASS_ID:
                people_count += 1
                psevdo_id += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'ID:{psevdo_id}. Confidence:{conf:.2f}'
                cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt >0:
                    fps = 1.0/dt

                cv2.putText(frame, f'People count:{people_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(frame, f'FPS:{fps:.1f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
