import cv2
import numpy as np
from PIL import Image

def predict_video(video_path, face_detector, image_model, frame_interval=5):
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
            # BGR -> RGB -> PIL.Image
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = face_detector.mtcnn.detect(img_pil)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = np.array(img_pil.crop((x1, y1, x2, y2)))

                    # 保證三通道
                    if len(face.shape) == 2:
                        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                    elif face.shape[2] == 4:
                        face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)

                    # resize 至模型輸入大小
                    face = cv2.resize(face, (224, 224))
                    probs.append(image_model.predict(face))

        idx += 1

    cap.release()
    return None if len(probs) == 0 else float(max(probs))
