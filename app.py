import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

from models.face_detector import FaceDetector
from models.image_model import DeepfakeImageModel
from models.video_inference import predict_video

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🎭 Deepfake Detector")
st.sidebar.markdown("""
Upload an image or video and the system will predict whether it is a Deepfake.
- **Supported image formats:** JPG, PNG  
- **Supported video format:** MP4
""")

# ===============================
# Load Models (cached)
# ===============================
@st.cache_resource
def load_all():
    # device='cpu' or 'cuda'
    return FaceDetector(device='cpu'), DeepfakeImageModel("saved_models/deepfake_model.pth")

face_detector, image_model = load_all()
image_model.model.eval()

# ===============================
# File Uploader
# ===============================
file = st.file_uploader("📁 Upload an image or video", type=["jpg", "png", "mp4"])

# ===============================
# Main Display
# ===============================
if file:
    if "image" in file.type:
        # --- 讀取圖片 ---
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        st.image(img[:, :, ::-1], caption="Uploaded Image", use_column_width=True)

        # --- OpenCV to PIL ---
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # --- 偵測人臉 ---
        face = face_detector.detect_faces_image(img_pil)
        if face is None:
            st.error("❌ No face detected.")
        else:
            # --- 推論 ---
            p = float(image_model.predict(face))

            # --- 卡片顯示結果 ---
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Prediction Result")
                st.metric("Fake Probability", f"{p*100:.2f}%")
            with col2:
                st.progress(p)

            # 顯示直觀顏色條
            st.markdown(f"""
            <div style="background-color:#ddd; width:100%; border-radius:10px; height:20px;">
                <div style="width:{p*100}%; background-color:#e63946; height:100%; border-radius:10px;"></div>
            </div>
            """, unsafe_allow_html=True)

    else:  # 影片
        # --- 暫存影片 ---
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        st.video(tfile.name, start_time=0)

        with st.spinner("🔍 Analyzing video..."):
            p = predict_video(tfile.name, face_detector, image_model)

        if p is None:
            st.error("❌ No face detected in video.")
        else:
            st.markdown("---")
            st.subheader("Prediction Result")
            st.metric("Fake Probability", f"{p*100:.2f}%")

            # 顯示進度條顏色
            st.progress(p)
            st.markdown(f"""
            <div style="background-color:#ddd; width:100%; border-radius:10px; height:20px;">
                <div style="width:{p*100}%; background-color:#e63946; height:100%; border-radius:10px;"></div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Please upload an image or video to start detection.")
