import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe FaceMesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0, 128))

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

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    # Preprocess the image
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d, face_2d = [], []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_px = []
            for landmark in face_landmarks.landmark:
                x_px = int(landmark.x * img_w)
                y_px = int(landmark.y * img_h)
                landmarks_px.append((x_px, y_px))
                
                # Draw face landmarks as pixels
                cv2.circle(image, (x_px, y_px), 1, (0, 255, 0), -1)  # Small circle (pixel)

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

            # Head Pose Estimation
            face_2d = np.array([landmarks_px[i] for i in [1, 33, 61, 263, 291, 199]], dtype=np.float64)
            face_3d = np.array([[x, y, landmark.z] for (x, y), landmark in zip(face_2d, face_landmarks.landmark)], dtype=np.float64)
            
            # Camera matrix and distortion coefficients
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display text information
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, f"x: {x:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"y: {y:.2f}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"z: {z:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image
    cv2.imshow('Head Pose and Mouth Opening Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
