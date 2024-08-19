# Image-Processing

##Overview

This project aims to classify the different growth stages of an orchid flower using a real-time webcam stream and an inference pipeline powered by Roboflow. The model identifies four distinct growth stages: Stage 1, Stage 2, Stage 3, and Petal. The project implements a real-time video inference system that processes the video stream from a webcam and overlays the predicted classifications and bounding boxes on the video feed.

##Features

1. Real-time orchid growth stage classification: Classifies orchid flowers into four stages using a pre-trained model on Roboflow.
2. Webcam Integration: Captures live video feed from a USB camera.
3. Flask-based server: Displays the classified output as a video stream through a web interface.
4. Bounding box rendering: Draws bounding boxes around detected growth stages with labels and confidence scores.

##Installation
Python 3.7+
cv2 (OpenCV)
flask
roboflow Python package
A Roboflow API key and model set up

##Setup Instructions

**Clone the Repository:**
```bash
git clone https://github.com/your-username/orchid-growth-classification.git
cd orchid-growth-classification
```

**Install Required Libraries:**
Make sure you have the necessary dependencies installed. You can install them using:
```bash
pip install -r requirements.txt
```

**Set up Environment Variables:**
Ensure you have your Roboflow API key, Workspace Name, Project Name, and Model Version Number ready.

**Run the Flask Server:**
After setup, start the server using:
```bash
python app.py
```

**Access the Web Interface:**
Open your web browser and go to:
```arduino
http://localhost:5000
````
to view the live video feed with growth stage classifications.
