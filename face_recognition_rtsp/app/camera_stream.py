import cv2
import asyncio
from multiprocessing import Process, Queue
import face_recognition
import numpy as np
import os
import torch
import time
import queue

class CameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.running = False
        self.frame = None
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_process_interval = 5
        self.last_frame = None
        self.face_data = {}
        self.processing_faces = False
        self.last_face_locations = []

        self.face_process = Process(
            target=CameraStream.face_detection_process,
            args=(self.frame_queue, self.result_queue, self.face_process_interval, self.face_data, self)
        )
        self.face_process.start()

    async def start(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.running = True
        asyncio.create_task(self._read_frames())

    async def _read_frames(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    self.frame_queue.put(frame)
                    if not self.result_queue.empty():
                        self.frame = self.result_queue.get()
                else:
                    print(f"Error: Could not read frame from {self.rtsp_url}.")
                    await asyncio.sleep(0.1)  # ลดเวลาหน่วงเมื่ออ่านเฟรมไม่ได้
            except Exception as e:
                print(f"Error reading frames from {self.rtsp_url}: {e}")
                await asyncio.sleep(0.1)  # ลดเวลาหน่วงเมื่อเกิดข้อผิดพลาด
            await asyncio.sleep(0.01)  # เพิ่มเฟรมเรต

    @staticmethod
    def face_detection_process(frame_queue, result_queue, face_process_interval, face_data, camera_stream):
        known_face_encodings = []
        known_face_names = []
        frame_count = 0

        for filename in os.listdir("database"):
            full_path = os.path.join("database", filename)
            if os.path.isdir(full_path):
                for image_file in os.listdir(full_path):
                    image = face_recognition.load_image_file(os.path.join(full_path, image_file))
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(filename)

        while True:
            try:
                frame = frame_queue.get(timeout=0.1)  # ลด timeout
            except queue.Empty:
                continue
            frame_count += 1

            face_locations = face_recognition.face_locations(frame)

            new_faces_detected = False
            if len(face_locations) != len(camera_stream.last_face_locations):
                new_faces_detected = True
            else:
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    last_top, last_right, last_bottom, last_left = camera_stream.last_face_locations[i]
                    if abs(top - last_top) > 10 or abs(right - last_right) > 10 or abs(bottom - last_bottom) > 10 or abs(left - last_left) > 10:  # ลดเกณฑ์การตรวจจับใบหน้าใหม่
                        new_faces_detected = True
                        break

            if new_faces_detected and not camera_stream.processing_faces:
                camera_stream.processing_faces = True
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

                face_data.clear()
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    face_data[(left, top, right, bottom)] = name
                camera_stream.processing_faces = False

            for (left, top, right, bottom), name in face_data.items():
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            result_queue.put(frame)
            camera_stream.last_face_locations = face_locations

    async def generate_frames(self):
        while True:
            if self.frame is not None:
                _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.01)  # เพิ่มเฟรมเรต

    async def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.face_process.terminate()