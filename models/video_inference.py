import cv2
import numpy as np

def predict_video(video_path, face_detector, image_model, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    probs = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            faces = face_detector.detect(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    probs.append(image_model.predict(face))
        idx += 1

    cap.release()
    return None if len(probs) == 0 else float(np.mean(probs))
