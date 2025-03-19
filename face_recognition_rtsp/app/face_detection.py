import cv2
from deepface import DeepFace

class FaceDetector:
    def __init__(self):
        self.detector_name = "SFace"
        self.detector_backend = "retinaface"
        self.detector_options = {'retinaface_use_cpu': True, 'retinaface_allow_upscaling': False}

    def detect_and_recognize_faces(self, frame):
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = DeepFace.find(img_path=small_frame, db_path="database", model_name=self.detector_name,
                                 detector_backend=self.detector_backend, enforce_detection=False, silent=True,
                                 detector_options=self.detector_options)
            if len(faces) > 0:
                for face_list in faces:
                    for index, instance in face_list.iterrows():
                        x, y, w, h = int(instance['facial_area']['x'] * 2), int(instance['facial_area']['y'] * 2), int(
                            instance['facial_area']['w'] * 2), int(instance['facial_area']['h'] * 2)
                        identity = instance['identity'].split('\\')[-1].split('.')[0]
                        confidence = instance['distance']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{identity} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)
        except ValueError as e:
            print(f"Error processing frame: {e}")
        return frame