import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

class FaceDetector:
    def __init__(self, min_confidence=0.6):
        self.detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_confidence
        )

    def detect(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)

        faces = []
        h, w, _ = image.shape

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                faces.append((x, y, bw, bh))

        return faces
