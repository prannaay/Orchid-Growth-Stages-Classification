from flask import Flask, render_template, Response
import cv2
import numpy as np
from roboflow import Roboflow

app = Flask(__name__)

# Initialize the USB camera
camera_index = 0  # Change this if you have multiple cameras
camera = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not camera.isOpened():
    raise RuntimeError("Error: Could not open USB camera")

# Initialize the Roboflow model
api_key = "API_KEY"  # Replace with your Roboflow API key
workspace_name = "WORKSPACE"  # Replace with your workspace name
project_name = "MODEL_NAME"  # Replace with your project name
version_number = VERSION_NUMBER  # Replace with your model version number

rf = Roboflow(api_key=api_key)
workspace = rf.workspace(workspace_name)
project = workspace.project(project_name)
model = project.version(version_number).model

import os
import tempfile

def process_frame_with_roboflow(frame):
    # Convert the frame to a format suitable for Roboflow inference
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    # Perform inference using Roboflow model
    try:
        # Predict using the file path
        prediction = model.predict(temp_file_path, confidence=40).json()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return frame  # Return the original frame in case of an error
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

    # Render bounding boxes and labels onto the frame
    for box in prediction['predictions']:
        x0 = int(box['x'] - box['width'] / 2)
        y0 = int(box['y'] - box['height'] / 2)
        x1 = int(box['x'] + box['width'] / 2)
        y1 = int(box['y'] + box['height'] / 2)
        
        # Draw the bounding box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # Put the label
        label = f"{box['class']} ({box['confidence']:.2f})"
        cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def generate_frames():
    while True:
        # Capture frame-by-frame from the USB camera
        ret, frame = camera.read()

        if not ret:
            print("Error: Failed to capture image")
            continue

        # Process the frame with Roboflow Inference
        frame = process_frame_with_roboflow(frame)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to be served to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
