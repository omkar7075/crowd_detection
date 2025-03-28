📌 AI-Powered Crowd Detection System
This project is an AI-based Crowd Detection System using YOLOv8 for object detection and DeepSORT for object tracking. The system can process images, provide a live camera feed, and count people in real-time.

🚀 Features
📷 Upload an image to detect and count people.

🎥 Real-time crowd detection via live camera feed.

📊 Tracks people using DeepSORT.

📡 Flask-based API with a frontend for easy usage.

📌 Installation & Setup
Follow the steps below to set up and run the project.

1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-repo/crowd-detection-ai.git
cd crowd-detection-ai
2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Download YOLOv8 Model
The script automatically downloads the YOLOv8 model (yolov8n.pt). If it fails, download it manually:

bash
Copy
Edit
wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
4️⃣ Run the Flask App
Start the Flask server:

bash
Copy
Edit
python app.py
It will run on http://127.0.0.1:5000/.

📌 Usage
🌟 1. Upload an Image
Open your browser and go to http://127.0.0.1:5000/.

Click on "Choose File", upload an image, and press "Detect Crowd".

The processed image with detected people will be displayed.

🌟 2. Real-Time Detection
The live camera feed will automatically update.

It will count and track people in real-time.

📌 API Endpoints
Endpoint	Method	Description
/	GET	Home page with upload & live feed UI
/upload	POST	Upload an image and detect people
/camera_feed	GET	Real-time camera detection feed
/live_count	GET	Get current detected people count
📌 Troubleshooting
🔹 Error: Missing YOLO model
→ Ensure yolov8n.pt exists in the models/ folder.

🔹 Error: Camera not detected
→ Check if another program is using the webcam.

🔹 Error: Flask app not running
→ Run python app.py and check for errors.

📌 Tech Stack
Python (Flask, OpenCV, NumPy, Requests)

Machine Learning (YOLOv8, DeepSORT)

Frontend (HTML, JavaScript, CSS)

📌 Contributing
Feel free to contribute! Fork the repo, create a branch, and submit a PR.

📌 License
🔹 MIT License - Free to use and modify.

