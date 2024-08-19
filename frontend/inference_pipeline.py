import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# Initialize the USB camera
camera_index = 0  # Change this if you have multiple cameras and need a different index
camera = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not camera.isOpened():
    raise RuntimeError("Error: Could not open USB camera")

# Initialize the InferencePipeline from Roboflow
inference_pipeline = InferencePipeline.init(
    model_id="Model_ID/Version",  # Replace with your model ID
    max_fps=0.5,
    confidence=0.1,
    video_reference="",  # Placeholder, will be set per frame
    on_prediction=render_boxes,
    api_key="API_KEY"  # Replace with your API key
)

# Start the InferencePipeline
inference_pipeline.start()

# Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
# receive the actual position of the bounding box on the image
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Function to process each frame with Roboflow Inference
def process_frame_with_roboflow(frame):
    # Convert the frame to a format suitable for Roboflow inference
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = np.frombuffer(buffer, dtype=np.uint8)
    
    # Create a temporary file to simulate a video frame input
    temp_file_path = 'temp_frame.jpg'  # Use a relative path for the temporary file
    with open(temp_file_path, 'wb') as f:
        f.write(encoded_frame)

    # Update the video_reference to the temporary file
    inference_pipeline.video_reference = temp_file_path

    # Process the temporary file with the inference pipeline
    inference_pipeline.process()

while True:
    # Capture frame-by-frame from the USB camera
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    if frame is not None:
        # Process the frame with Roboflow Inference
        process_frame_with_roboflow(frame)

        # Show the frame
        cv2.imshow("preview", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Stop the InferencePipeline
inference_pipeline.stop()
