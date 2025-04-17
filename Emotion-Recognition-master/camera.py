import cv2
from model import ERModel
import numpy as np


class VideoCamera(object):
    def __init__(self):
        # Load the face detection model
        self.facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load the emotion recognition model
        self.model = ERModel("model.json", "model_weights.h5")

        # Set font for text overlay
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Initialize camera
        self.video = cv2.VideoCapture(0)

        # Ensure the camera is opened correctly
        if not self.video.isOpened():
            raise IOError("Cannot open webcam")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, fr = self.video.read()
        if not success:
            return None

        # Convert to grayscale
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract face region
            fc = gray_fr[y:y + h, x:x + w]

            # Resize for model input
            roi = cv2.resize(fc, (48, 48))

            # Debug information
            print(f"Face ROI shape: {roi.shape}, min: {np.min(roi)}, max: {np.max(roi)}")

            # Try different preprocessing approaches
            # Approach 1: Original code's approach
            pred_input = roi[np.newaxis, :, :, np.newaxis]

            # Print the predictions to debug
            predictions = self.model.loaded_model.predict(pred_input, verbose=0)
            pred_class = np.argmax(predictions)
            pred_confidence = np.max(predictions) * 100
            emotion = self.model.EMOTIONS_LIST[pred_class]

            print(f"Predictions: {predictions}")
            print(f"Detected emotion: {emotion} with {pred_confidence:.2f}% confidence")

            # Draw result on frame
            label = f"{emotion}: {pred_confidence:.1f}%"
            cv2.putText(fr, label, (x, y - 10), self.font, 0.7, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()