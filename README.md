
# 📌 AI-Powered Crowd Detection System
This project is an AI-based Crowd Detection System using YOLOv8 for object detection and DeepSORT for object tracking. The system can process images, provide a live camera feed, and count people in real-time.

# 🚀 Features

📷 Upload an image to detect and count people.

🎥 Real-time crowd detection via live camera feed.

📊 Tracks people using DeepSORT.

📡 Flask-based API with a frontend for easy usage.

# 📌 Prerequisites

Before starting, ensure you have:

✔️ Python 3.8+ installed

✔️ pip and virtualenv installed

```
pip install --upgrade pip
pip install virtualenv
```

1️⃣ Setup Virtual Environment

Navigate to the project directory and create a virtual environment:

```
cd crowd_detection
virtualenv venv
```
For Windows use:

```
python3 -m venv venv
```

Activate the Virtual Environment
Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

You should see (venv) in the terminal, indicating activation.

# 📌 Installation & Setup
Follow the steps below to set up and run the project.

1️⃣ Clone the Repository
```
git clone https://github.com/omkar7075/crowd_detection.git
cd crowd_detection
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Download YOLOv8 Model
The script automatically downloads the YOLOv8 model (yolov8n.pt). If it fails, download it manually:
```
wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```
4️⃣ Run the Flask App
Start the Flask server:
```
python app.py
```
# 📌 Usage
🌟 1. Upload an Image

Open your browser and go to http://127.0.0.1:5000/.

Click on "Choose File", upload an image, and press "Detect Crowd".

The processed image with detected people will be displayed.

🌟 2. Real-Time Detection

The live camera feed will automatically update.

It will count and track people in real-time.

# 📌 Tech Stack
Python (Flask, OpenCV, NumPy, Requests)

Machine Learning (YOLOv8, DeepSORT)

Frontend (HTML, JavaScript, CSS)

# 📌 Contributing
Feel free to contribute! Fork the repo, create a branch, and submit a PR.

# 📌 License
🔹 MIT License - Free to use and modify.

