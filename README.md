pip install torch torchvision opencv-python-headless face_recognition_models onnxruntime-gpu fastapi uvicorn numpy

pip install -r requirements.txt

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

http://127.0.0.1:8000
