import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

from models.face_detector import FaceDetector
from models.image_model import DeepfakeImageModel
from models.video_inference import predict_video

st.set_page_config(page_title="Deepfake Detector")
st.title("🎭 Deepfake Image & Video Detector")

@st.cache_resource
def load_all():
    # device='cpu' or 'cuda' 可依環境調整
    return FaceDetector(device='cpu'), DeepfakeImageModel("saved_models/deepfake_model.pth")

face_detector, image_model = load_all()
image_model.model.eval()

file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4"])

if file:
    if "image" in file.type:
        # 讀取圖片
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 將 BGR 轉 RGB，給 Streamlit 顯示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 將 OpenCV BGR 轉 PIL.Image
        img_pil = Image.fromarray(img_rgb)

        # 使用 MTCNN 偵測人臉
        faces = face_detector.mtcnn.detect(img_pil)[0]  # boxes
        if faces is None:
            st.error("No face detected.")
        else:
            # 在原圖上畫 bounding box
            for box in faces:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 紅色框

            st.image(img_rgb)  # 顯示帶框的圖片

            # 取第一張臉裁切做推論
            x1, y1, x2, y2 = map(int, faces[0])
            face_crop = img_rgb[y1:y2, x1:x2]
            p = float(image_model.predict(face_crop))
            st.success(f"Fake Probability: {p:.2f}")
            st.progress(p)
            
    else:  # 影片
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        st.video(tfile.name)

        with st.spinner("Analyzing..."):
            # 使用 MTCNN 偵測影片人臉，並丟進模型
            p = predict_video(tfile.name, face_detector, image_model)

        if p is None:
            st.error("No face detected.")
        else:
            st.success(f"Fake Probability: {p*100: .2f}%")
            st.progress(p)
