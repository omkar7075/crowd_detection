import os
import cv2
import numpy as np
import requests
import subprocess
from flask import Flask, request, render_template, jsonify, send_from_directory, Response
from flask_cors import CORS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Setup Directories ---
PROJECT_DIR = os.path.abspath("crowd_detection_ai")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
UPLOADS_DIR = os.path.join(PROJECT_DIR, "uploads")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")

for directory in [MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, TEMPLATES_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Install Dependencies ---
try:
    import ultralytics
    import flask
except ImportError:
    subprocess.run(["pip", "install", "ultralytics", "flask", "flask-cors", "opencv-python", "deep_sort_realtime"])

# --- Download YOLOv8 Model ---
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    response = requests.get(YOLO_URL, stream=True)
    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

# --- Initialize YOLOv8 and DeepSORT ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

deep_sort = DeepSort(max_age=30)

# --- Flask App Setup ---
app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOADS_DIR
app.config["OUTPUT_FOLDER"] = OUTPUTS_DIR

# --- Detection Functions ---

def detect_and_track(filepath):
    """Processes an image for crowd detection and tracking."""
    image = cv2.imread(filepath)
    if image is None:
        return None, "Invalid image file"

    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()
    detections = np.atleast_2d(detections)

    processed_detections = []
    for det in detections:
        if det.shape[0] < 6:
            continue
        x_min, y_min, x_max, y_max, conf, class_id = det
        if int(class_id) != 0:
            continue
        width, height = x_max - x_min, y_max - y_min
        processed_detections.append([[float(x_min), float(y_min), float(width), float(height)], float(conf)])

    if not processed_detections:
        cv2.putText(image, "No person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        processed_path = os.path.join(app.config["OUTPUT_FOLDER"], os.path.basename(filepath))
        cv2.imwrite(processed_path, image)
        return processed_path, []

    tracks = deep_sort.update_tracks(processed_detections, frame=image)

    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        track_id = track.track_id
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_path = os.path.join(app.config["OUTPUT_FOLDER"], os.path.basename(filepath))
    cv2.imwrite(processed_path, image)
    return processed_path, tracks

def generate_frames():
    """Live camera feed with YOLOv8 detection and DeepSORT tracking."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()
        detections = np.atleast_2d(detections)

        processed_detections = []
        for det in detections:
            if det.shape[0] < 6:
                continue
            x_min, y_min, x_max, y_max, conf, class_id = det
            if int(class_id) != 0:
                continue
            width, height = x_max - x_min, y_max - y_min
            processed_detections.append([[float(x_min), float(y_min), float(width), float(height)], float(conf)])

        tracks = deep_sort.update_tracks(processed_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_ltrb()
            track_id = track.track_id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    processed_path, tracks = detect_and_track(filepath)

    if processed_path is None:
        return jsonify({"error": tracks}), 400

    total_people = len(tracks)
    detections_info = [{"id": int(track.track_id)} for track in tracks if track.is_confirmed()]
    processed_url = f"http://127.0.0.1:5000/outputs/{os.path.basename(processed_path)}"

    return jsonify({
        "message": "Detection completed",
        "processed_image": processed_url,
        "total_people_detected": total_people,
        "detections": detections_info
    }), 200

@app.route("/camera_feed")
def camera_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live_count")
def live_count():
    return jsonify({"current_count": len(deep_sort.tracker.tracks)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
