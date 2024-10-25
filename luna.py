import cv2
from ultralytics import YOLO
import numpy as np
import time
import os

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
}

def draw_bounding_boxes(frame, results):
    object_count = {}
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())

            if conf > 0.5:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{model.names[cls]} {conf:.2f}"
                color = color_map.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if model.names[cls] in object_count:
                    object_count[model.names[cls]] += 1
                else:
                    object_count[model.names[cls]] = 1

    return object_count

def main():
    cv2.namedWindow('YOLOv8 Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 Object Detection', 1280, 720)

    if not os.path.exists('detected_frames'):
        os.makedirs('detected_frames')

    while True:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            continue

        results = model(frame)
        object_count = draw_bounding_boxes(frame, results)

        cv2.putText(frame, f'Objects Detected: {len(results[0].boxes)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(f'detected_frames/frame_{timestamp}.jpg', frame)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('YOLOv8 Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

