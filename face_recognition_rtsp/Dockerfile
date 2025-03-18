# ใช้ base image ที่มี Python 3.9
FROM python:3.9-slim-buster

# ติดตั้ง dependencies ของระบบปฏิบัติการ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    cmake \
    build-essential \
    libx11-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# สร้าง working directory
WORKDIR /app

# คัดลอก requirements.txt และติดตั้ง dependencies ของ Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดไปยัง working directory
COPY . .

# เปิด port 8000
EXPOSE 8000

# รันแอปพลิเคชันด้วย uvicorn โดยระบุ app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]