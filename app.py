import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from models.face_detector import FaceDetector
from models.image_model import DeepfakeImageModel
from models.video_inference import predict_video
from models.grad_cam import GradCAM, overlay_cam


# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_all():
    return FaceDetector(device='cpu'), DeepfakeImageModel("saved_models/deepfake_model.pth")

face_detector, image_model = load_all()
image_model.model.eval()

target_layer = image_model.model.features[-1]
grad_cam = GradCAM(image_model.model, target_layer)

# ===============================
# Header
# ===============================
st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#e63946;">🎭 Deepfake Image & Video Detector</h1>
        <p style="font-size:18px; color:#555;">
        Upload an image or video. The system will detect if it is Deepfake and show probability.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# ===============================
# Upload Section
# ===============================
# file = st.file_uploader("📁 Upload an image or video", type=["jpg","png","mp4"], width=500)

# 用 st.columns 置中
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.write("")

with col2:
    file = st.file_uploader("📁 Upload an image or video", type=["jpg","png","mp4"], width=500)

with col3:
    st.write("")

def draw_face_box(img_pil, box):
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle(box, outline="#e63946", width=4)
    return img_pil

# ===============================
# Main Processing
# ===============================
if file:
    if "image" in file.type:
        # Read Image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Detect faces
        face, box = face_detector.detect_faces_image(img_pil, return_box=True)
        if face is None:
            st.error("❌ No face detected.")
        else:
            # Draw face box
            img_pil = draw_face_box(img_pil, box)
            # st.image(img_pil, caption="Detected Face", use_column_width=True)

            # Predict
            p = float(image_model.predict(face))

            image_model.model.zero_grad()

            # numpy -> torch tensor
            input_tensor = torch.from_numpy(face).float()

            # 若是 HWC 轉成 CHW（保險起見）
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.permute(2, 0, 1)

            input_tensor = input_tensor.unsqueeze(0)  # (1, C, H, W)

            cam = grad_cam.generate(input_tensor)
            cam_overlay = overlay_cam(img_pil, cam)


            # Display result in card style
            # 左右併排
            # 左右併排並置中
        outer_left, outer_center, outer_right = st.columns([1, 3, 1])

        with outer_center:
            img_col, col_empty, cam_col, col_empty2, prob_col = st.columns([1, 1, 1, 1, 1])

            card_width = 280

            with img_col:
                st.markdown(
                    "<div style='display:flex; justify-content:center;'>",
                    unsafe_allow_html=True
                )
                st.image(img_pil, caption="Detected Face", width=card_width)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_empty:
                st.write("")

            with cam_col:
                st.markdown(
                    "<div style='display:flex; justify-content:center;'>",
                    unsafe_allow_html=True
                )
                st.image(cam_overlay, caption="Grad-CAM Explanation", width=card_width)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_empty2:
                st.write("")

            with prob_col:
                st.markdown(f"""
                <div style="
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    height:100%;
                ">
                    <h3 style="color:#e63946; margin-bottom:5px;">Fake Probability</h3>
                    <div style="width:220px; height:220px;">
                        <svg viewBox="0 0 36 36" class="circular-chart">
                            <path class="circle-bg"
                                d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0-31.831"/>
                            <path class="circle"
                                stroke-dasharray="{p*100}, 100"
                                d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0-31.831"/>
                            <text x="18" y="18" class="percentage">{p*100:.1f}%</text>
                        </svg>
                    </div>
                </div>

                <style>
                .circular-chart {{ width:100%; height:100%; }}
                .circle-bg {{ fill:none; stroke:#eee; stroke-width:4; }}
                .circle {{ fill:none; stroke:#e63946; stroke-width:4; stroke-linecap:round; }}
                .percentage {{ fill:#e63946; font-size:0.6em; font-weight:bold;
                            text-anchor:middle; dominant-baseline:middle; }}
                </style>
                """, unsafe_allow_html=True)


    else:  # Video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        # Predict probability
        with st.spinner("🔍 Analyzing video..."):
            p = predict_video(tfile.name, face_detector, image_model)

        if p is None:
            st.error("❌ No face detected in video.")
        else:
            # Layout: outer columns for centering
            col_left, col_center, col_right = st.columns([1,3,1])
            with col_center:
                inner_col_video, inner_col_result = st.columns([1,1])
                vid_width = 500
                with inner_col_video:
                    st.video(tfile.name, start_time=0, format="video/mp4", width=vid_width)
                with inner_col_result:
                    st.markdown(f"""
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
                        <h2 style="color:#e63946; margin-bottom:5px;">Fake Probability</h2>
                        <div style="position: relative; width:250px; height:250px; margin-top:5px;">
                            <svg viewBox="0 0 36 36" class="circular-chart">
                                <path class="circle-bg"
                                    d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0-31.831"/>
                                <path class="circle"
                                    stroke-dasharray="{p*100}, 100"
                                    d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0-31.831"/>
                                <text x="18" y="18" class="percentage">{p*100:.1f}%</text>
                            </svg>
                        </div>
                    </div>
                    <style>
                    .circular-chart {{ display:block; width:100%; height:100%; }}
                    .circle-bg {{ fill:none; stroke:#eee; stroke-width:4; }}
                    .circle {{ fill:none; stroke:#e63946; stroke-width:4; stroke-linecap:round; transition: stroke-dasharray 0.3s; }}
                    .percentage {{ fill:#e63946; font-size:0.6em; font-weight:bold; text-anchor:middle; dominant-baseline:middle; }}
                    </style>
                    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Please upload an image or video to start detection.")
