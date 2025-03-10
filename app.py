from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = r"C:\Users\rohin\chatgpt_yolo\static"
OUTPUT_FOLDER = r"C:\Users\rohin\chatgpt_yolo\static\output_videos"
MODEL_PATH = "models/best.pt"  # Ensure this file exists
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded."

        file = request.files["file"]

        if file.filename == "":
            return "No selected file."

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"processed_{filename}")

            file.save(input_path)

            # Process the video
            process_video(input_path, output_path)

            return render_template("index.html", processed_video=output_path)

    return render_template("index.html", processed_video=None)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            class_names = ["gun", "knife"]  # Adjust according to your dataset
            label = class_names[class_id] if class_id < len(class_names) else "unknown"
            color = (0, 0, 255) if label == "gun" else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
