# Image-Processing

## Overview

This project aims to classify the different growth stages of an orchid flower using a real-time webcam stream and an inference pipeline powered by Roboflow. The model identifies four distinct growth stages: Stage 1, Stage 2, Stage 3, and Petal. The project implements a real-time video inference system that processes the video stream from a webcam and overlays the predicted classifications and bounding boxes on the video feed.

## Features

1. Real-time orchid growth stage classification: Classifies orchid flowers into four stages using a pre-trained model on Roboflow.
2. Webcam Integration: Captures live video feed from a USB camera.
3. Flask-based server: Displays the classified output as a video stream through a web interface.
4. Bounding box rendering: Draws bounding boxes around detected growth stages with labels and confidence scores.

## Installation
- Python 3.7+
- cv2 (OpenCV)
- flask
- roboflow Python package
- A Roboflow API key and model set up

## Prerequisites
Roboflow Model Training
Before running the code, you need to train a model on Roboflow. This involves:

1. Data Collection: Gather images of orchids in various growth stages.
2. Annotation: Label the images according to the four growth stages: Stage 1, Stage 2, Stage 3, and Petal.
3. Model Training: Use Roboflow's platform to train a model on the annotated dataset.
4. Deployment: Once trained, deploy the model and obtain the API key and model details.

After completing the training process, you can integrate the model into this project to classify orchid growth stages in real-time.

## Setup Instructions

**1. Clone the Repository:**
```bash
git clone https://github.com/your-username/orchid-growth-classification.git
cd orchid-growth-classification
```

**2. Install Required Libraries:**
Make sure you have the necessary dependencies installed. You can install them using:
```bash
pip install -r requirements.txt
```

**3. Set up Environment Variables:**
Ensure you have your Roboflow API key, Workspace Name, Project Name, and Model Version Number ready.

**4. Run the Flask Server:**
After setup, start the server using:
```bash
python app.py
```

**5. Access the Web Interface:**
Open your web browser and go to:
```arduino
http://localhost:5000
````
to view the live video feed with growth stage classifications.
