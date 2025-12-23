import cv2
import numpy as np
from PIL import Image

def predict_video(video_path, face_detector, image_model, frame_interval=15):
    """
    video_path: 影片路徑
    face_detector: MTCNN 版本 FaceDetector
    image_model: DeepfakeImageModel
    frame_interval: 每隔幾幀抽取一次
    """
    cap = cv2.VideoCapture(video_path)
    probs = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            # 將 BGR -> RGB -> PIL
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = face_detector.detect_faces_image(img_pil)

            # detect_faces_image 返回單張裁切人臉 (numpy array)
            if faces is not None:
                probs.append(image_model.predict(faces))
        idx += 1

    cap.release()
    return None if len(probs) == 0 else float(np.max(probs))
