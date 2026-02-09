import cv2
import os
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')
INPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'output_video.mp4')

USE_WEBCAM = True

if USE_WEBCAM:
    source = 0
else:
    source = INPUT_VIDEO_PATH

MODEL_PATH = 'yolov8n.pt'

CONF_TRESH = 0.5

TRACKER = 'bytetrack.yaml'

SAVE_VIDEO = True

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


seen_id_total = set()

seen_id_class = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model.track(frame, conf = CONF_TRESH, tracker = TRACKER, persist = True, verbose = True)
    r = result[0]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow('frame', frame)

        if writer is not None:
            writer.write(frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes

    xyxy = boxes.xyxy.cpu().numpy()

    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    track_id = None
    if boxes.id is not None:
        track_id = boxes.id.cpu().numpy()

    for i in range(len(xyxy)):
        x1, x2, y1, y2 = xyxy[i].astype(int)
        class_id = int(cls[i])

        class_name = model.names[class_id]

        score = conf[i]

        tid = -1

        if track_id is not None:
            tid = int(track_id[i])


        if tid != -1:
            seen_id_total.add(tid)

            if class_name not in seen_id_class:
                seen_id_class[class_name] = set()

            seen_id_class[class_name].add(tid)

        label = (f'{class_name} {score:.2f}')
        if tid != -1:
            label += f'ID {tid}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        (tw, th), _ = cv2. getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0), thickness = 2)


        total = len(seen_id_total)
        cv2.putText(frame, f'unique objects {total}', (10, 30), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)

        cv2.imshow('frame', frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

