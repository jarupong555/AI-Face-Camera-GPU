from fastapi import FastAPI, Request, Form, HTTPException, responses
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.camera_stream import CameraStream
from app.face_detection import FaceDetector
import asyncio
from typing import List

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

camera_streams = {}
face_detectors = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """หน้าแรกของเว็บแอปพลิเคชัน"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/add_cameras")
async def add_cameras(request: Request, num_cameras: int = Form(...), ips: List[str] = Form(...), users: List[str] = Form(...), passwords: List[str] = Form(...)):
    """เพิ่มกล้องวงจรปิดไปยังระบบ"""
    if len(ips) != num_cameras or len(users) != num_cameras or len(passwords) != num_cameras:
        raise HTTPException(status_code=400, detail="Number of IPs, users, and passwords must match the number of cameras.")

    for i in range(num_cameras):
        rtsp_url = f"rtsp://{users[i]}:{passwords[i]}@{ips[i]}:554/cam/realmonitor?channel=1&subtype=0"
        camera_streams[ips[i]] = CameraStream(rtsp_url)
        face_detectors[ips[i]] = FaceDetector()  # สร้าง FaceDetector สำหรับแต่ละกล้อง
        await camera_streams[ips[i]].start()

    return responses.RedirectResponse(f"/camera/{ips[0]}", status_code=302)  # redirect ไปยังหน้าสตรีม

@app.get("/camera/{camera_ip}", response_class=HTMLResponse)
async def camera(request: Request, camera_ip: str):
    """แสดงหน้าสตรีมสำหรับกล้องที่ระบุ"""
    if camera_ip not in camera_streams:
        raise HTTPException(status_code=404, detail="Camera not found")

    return templates.TemplateResponse("camera.html", {"request": request, "camera_ip": camera_ip})

@app.get("/stream/{camera_ip}")
async def stream(camera_ip: str):
    """สตรีมวิดีโอจากกล้องที่ระบุ"""
    if camera_ip not in camera_streams:
        raise HTTPException(status_code=404, detail="Camera not found")

    frame_generator = camera_streams[camera_ip].generate_frames()
    return StreamingResponse(frame_generator, media_type="multipart/x-mixed-replace; boundary=frame")