from deepface import DeepFace
import cv2

class FaceDetector:
    def __init__(self):
        self.detector_name = "SFace"

    def detect_and_recognize_faces(self, frame):
        try:
            faces = DeepFace.find(img_path=frame, db_path="database", model_name=self.detector_name, detector_backend="mtcnn", enforce_detection=False, silent=True)
            if len(faces) > 0 and len(faces[0]) > 0:
                for index, instance in faces[0].iterrows():
                    x, y, w, h = int(instance['facial_area']['x']), int(instance['facial_area']['y']), int(instance['facial_area']['w']), int(instance['facial_area']['h'])
                    identity = instance['identity'].split('\\')[-1].split('.')[0]
                    confidence = instance['distance']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{identity} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except ValueError as e:
            print(f"Error processing frame: {e}")
        return frame