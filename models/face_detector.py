# face_detector.py
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, device='cpu'):
        """
        初始化 MTCNN
        device: 'cpu' 或 'cuda'，根據你的環境
        keep_all=True 可以偵測多張臉
        """
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect_faces_image(self, img_path, return_box=False):
        """
        偵測單張影像人臉並裁切
        img_path: 影像路徑或 PIL.Image 對象
        return_box: 是否回傳 bounding box
        回傳：
            return_box=False: np.array(cropped face) 或 None
            return_box=True: (np.array(cropped face), box) 或 None
        """
        if isinstance(img_path, str):
            img = Image.open(img_path).convert('RGB')
        elif isinstance(img_path, Image.Image):
            img = img_path.convert('RGB')
        else:
            raise ValueError("img_path must be a path or PIL.Image")

        boxes, _ = self.mtcnn.detect(img)
        if boxes is None:
            return None

        # 只取第一張臉
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img.crop((x1, y1, x2, y2))

        if return_box:
            return np.array(cropped), (x1, y1, x2, y2)
        else:
            return np.array(cropped)


    def detect_faces_video(self, video_path, max_frames=None, fps_sample=5):
        """
        偵測影片每幀人臉並裁切
        video_path: 影片路徑
        max_frames: 最多取多少幀（None = 全部）
        fps_sample: 每秒取幾幀
        回傳：list of numpy arrays（裁切後的臉）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, int(video_fps / fps_sample))

        faces = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, _ = self.mtcnn.detect(img)
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        faces.append(np.array(img.crop((x1, y1, x2, y2))))
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        return faces  # 可能是空列表 if no faces found

# --------------------------
# 以下為測試用
if __name__ == "__main__":
    detector = FaceDetector(device='cpu')

    # 單張影像測試
    img_face = detector.detect_faces_image("test_image.jpg")
    if img_face is not None:
        print("Image face detected:", img_face.shape)
    else:
        print("No face detected in image.")

    # 影片測試
    faces = detector.detect_faces_video("test_video.mp4", max_frames=50)
    print(f"{len(faces)} faces detected in video.")
