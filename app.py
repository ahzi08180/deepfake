import streamlit as st
import cv2
import tempfile
import numpy as np

from models.face_detector import FaceDetector
from models.image_model import DeepfakeImageModel
from models.video_inference import predict_video

st.set_page_config(page_title="Deepfake Detector")
st.title("🎭 Deepfake Image & Video Detector")

@st.cache_resource
def load_all():
    return FaceDetector(), DeepfakeImageModel("saved_models/demo_model.pth")

face_detector, image_model = load_all()

file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4"])

if file:
    if "image" in file.type:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.image(img[:, :, ::-1])

        faces = face_detector.detect(img)
        if not faces:
            st.error("No face detected.")
        else:
            probs = []
            for (x, y, w, h) in faces:
                probs.append(image_model.predict(img[y:y+h, x:x+w]))
            p = float(np.mean(probs))
            st.success(f"Fake Probability: {p:.2f}")
            st.progress(p)

    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        st.video(tfile.name)

        with st.spinner("Analyzing..."):
            p = predict_video(tfile.name, face_detector, image_model)

        if p is None:
            st.error("No face detected.")
        else:
            st.success(f"Fake Probability: {p:.2f}")
            st.progress(p)
