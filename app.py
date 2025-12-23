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
        st.image(img[:, :, ::-1])

        # 將 OpenCV BGR 轉 PIL.Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 使用 MTCNN 偵測人臉
        face = face_detector.detect_faces_image(img_pil)
        if face is None:
            st.error("No face detected.")
        else:
            # 推論
            p = float(image_model.predict(face))
            st.success(f"Fake Probability: {p:.2f*100}%")
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
            st.success(f"Fake Probability: {p:.2f*100}%")
            st.progress(p)
