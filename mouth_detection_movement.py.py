import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0, 128))  # Add alpha channel for transparency

# Calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Parameters for mouth opening detection
threshold = 20  # Adjust this threshold based on your use case
cheating_duration_threshold = 2.0  # Time in seconds for considering oral cheating
cheating_frequency_threshold = 3  # Number of times mouth opened within a period to consider cheating
cheating_period = 10  # Period in seconds to count the number of mouth openings

# Variables to track mouth opening
mouth_open_start_time = None
mouth_open_count = 0
mouth_open_times = []

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect the face mesh
    results = face_mesh.process(image_rgb)
    
    # Draw the face mesh annotations on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert facial landmarks to pixel coordinates
            landmarks_px = []
            for landmark in face_landmarks.landmark:
                x_px = int(landmark.x * image.shape[1])
                y_px = int(landmark.y * image.shape[0])
                landmarks_px.append((x_px, y_px))
            
            # Get coordinates of the mouth landmarks
            mouth_top_px = landmarks_px[13]  # Upper lip top
            mouth_bottom_px = landmarks_px[14]  # Lower lip bottom
            
            # Calculate distance between mouth top and bottom
            distance = calculate_distance(mouth_top_px, mouth_bottom_px)
            
            current_time = time.time()

            # Check if the mouth is open
            if distance > threshold:
                if mouth_open_start_time is None:
                    mouth_open_start_time = current_time
                else:
                    duration = current_time - mouth_open_start_time
                    if duration > cheating_duration_threshold:
                        cv2.putText(image, 'Oral Cheating Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                if mouth_open_start_time is not None:
                    duration = current_time - mouth_open_start_time
                    if duration > cheating_duration_threshold:
                        cv2.putText(image, 'Oral Cheating Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    mouth_open_times.append(current_time)
                    mouth_open_start_time = None

            # Remove mouth open times outside the cheating period
            mouth_open_times = [t for t in mouth_open_times if current_time - t <= cheating_period]
            
            # Check frequency of mouth openings
            if len(mouth_open_times) > cheating_frequency_threshold:
                cv2.putText(image, 'Frequent Oral Cheating Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Mouth Opening Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()