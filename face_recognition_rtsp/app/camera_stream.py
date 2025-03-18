import cv2
import asyncio
from multiprocessing import Process, Queue
import face_recognition
import numpy as np
import os
import torch

class CameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.running = False
        self.frame = None
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.face_process = Process(target=self.face_detection_process, args=(self.frame_queue, self.result_queue))
        self.face_process.start()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def start(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.running = True
        asyncio.create_task(self._read_frames())

    async def _read_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.frame_queue.put(frame)
                if not self.result_queue.empty():
                    self.frame = self.result_queue.get()
            await asyncio.sleep(0.03)

    def face_detection_process(self, frame_queue, result_queue):
        known_face_encodings = []
        known_face_names = []

        for filename in os.listdir("database"):
            full_path = os.path.join("database", filename)
            if os.path.isdir(full_path):
                for image_file in os.listdir(full_path):
                    image = face_recognition.load_image_file(os.path.join(full_path, image_file))
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(filename)

        while True:
            frame = frame_queue.get()
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

            result_queue.put(frame)

    async def generate_frames(self):
        while True:
            if self.frame is not None:
                _, buffer = cv2.imencode('.jpg', self.frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.03)

    async def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.face_process.terminate()