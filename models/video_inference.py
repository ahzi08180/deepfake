import cv2
import numpy as np
from PIL import Image

def predict_video(video_path, face_detector, image_model, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    probs = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = face_detector.detect_faces_image(img_pil)

            if faces is not None:
                for face in faces:
                    # resize 保證 224x224
                    face = cv2.resize(face, (224,224))
                    probs.append(image_model.predict(face))

        idx += 1

    cap.release()
    return None if len(probs) == 0 else float(max(probs))
