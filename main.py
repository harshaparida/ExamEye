# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # import time
# # #
# # # # Initialize MediaPipe FaceMesh and Drawing utilities
# # # mp_face_mesh = mp.solutions.face_mesh
# # # face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# # # mp_drawing = mp.solutions.drawing_utils
# # #
# # # # Define drawing specifications
# # # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0, 128))
# # #
# # # # Calculate Euclidean distance
# # # def calculate_distance(point1, point2):
# # #     return np.linalg.norm(np.array(point1) - np.array(point2))
# # #
# # # # Parameters for mouth opening detection
# # # threshold = 20  # Adjust this threshold based on your use case
# # # cheating_duration_threshold = 2.0  # Time in seconds for considering oral cheating
# # # cheating_frequency_threshold = 3  # Number of times mouth opened within a period to consider cheating
# # # cheating_period = 10  # Period in seconds to count the number of mouth openings
# # #
# # # # Variables to track mouth opening
# # # mouth_open_start_time = None
# # # mouth_open_count = 0
# # # mouth_open_times = []
# # #
# # # # Start video capture
# # # cap = cv2.VideoCapture(0)
# # #
# # # while cap.isOpened():
# # #     success, image = cap.read()
# # #     if not success:
# # #         break
# # #
# # #     start = time.time()
# # #
# # #     # Preprocess the image
# # #     image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
# # #     image.flags.writeable = False
# # #     results = face_mesh.process(image_rgb)
# # #     image.flags.writeable = True
# # #     image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
# # #
# # #     img_h, img_w, img_c = image.shape
# # #     face_3d, face_2d = [], []
# # #
# # #     if results.multi_face_landmarks:
# # #         for face_landmarks in results.multi_face_landmarks:
# # #             landmarks_px = []
# # #             for landmark in face_landmarks.landmark:
# # #                 x_px = int(landmark.x * img_w)
# # #                 y_px = int(landmark.y * img_h)
# # #                 landmarks_px.append((x_px, y_px))
# # #
# # #                 # Draw face landmarks as pixels
# # #                 cv2.circle(image, (x_px, y_px), 1, (0, 255, 0), -1)  # Small circle (pixel)
# # #
# # #             # Get coordinates of the mouth landmarks
# # #             mouth_top_px = landmarks_px[13]  # Upper lip top
# # #             mouth_bottom_px = landmarks_px[14]  # Lower lip bottom
# # #
# # #             # Calculate distance between mouth top and bottom
# # #             distance = calculate_distance(mouth_top_px, mouth_bottom_px)
# # #             current_time = time.time()
# # #
# # #             # Check if the mouth is open
# # #             if distance > threshold:
# # #                 if mouth_open_start_time is None:
# # #                     mouth_open_start_time = current_time
# # #                 else:
# # #                     duration = current_time - mouth_open_start_time
# # #                     if duration > cheating_duration_threshold:
# # #                         cv2.putText(image, 'Oral Cheating Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
# # #             else:
# # #                 if mouth_open_start_time is not None:
# # #                     duration = current_time - mouth_open_start_time
# # #                     if duration > cheating_duration_threshold:
# # #                         cv2.putText(image, 'Oral Cheating Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
# # #                     mouth_open_times.append(current_time)
# # #                     mouth_open_start_time = None
# # #
# # #             # Remove mouth open times outside the cheating period
# # #             mouth_open_times = [t for t in mouth_open_times if current_time - t <= cheating_period]
# # #
# # #             # Check frequency of mouth openings
# # #             if len(mouth_open_times) > cheating_frequency_threshold:
# # #                 cv2.putText(image, 'Frequent Oral Cheating Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
# # #
# # #             # Head Pose Estimation
# # #             face_2d = np.array([landmarks_px[i] for i in [1, 33, 61, 263, 291, 199]], dtype=np.float64)
# # #             face_3d = np.array([[x, y, landmark.z] for (x, y), landmark in zip(face_2d, face_landmarks.landmark)], dtype=np.float64)
# # #
# # #             # Camera matrix and distortion coefficients
# # #             focal_length = 1 * img_w
# # #             cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
# # #             dist_matrix = np.zeros((4, 1), dtype=np.float64)
# # #
# # #             # Solve PnP
# # #             success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
# # #             rmat, _ = cv2.Rodrigues(rot_vec)
# # #             angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
# # #
# # #             x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
# # #
# # #             if y < -10:
# # #                 text = "Looking Left"
# # #             elif y > 10:
# # #                 text = "Looking Right"
# # #             elif x < -10:
# # #                 text = "Looking Down"
# # #             elif x > 10:
# # #                 text = "Looking Up"
# # #             else:
# # #                 text = "Forward"
# # #
# # #             # Display text information
# # #             cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
# # #             cv2.putText(image, f"x: {x:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# # #             cv2.putText(image, f"y: {y:.2f}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# # #             cv2.putText(image, f"z: {z:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# # #
# # #     # Display the image
# # #     cv2.imshow('Head Pose and Mouth Opening Detection', image)
# # #
# # #     if cv2.waitKey(5) & 0xFF == 27:
# # #         break
# # #
# # # cap.release()
# # # cv2.destroyAllWindows()
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time
# # import torch
# #
# # def run_proctoring_system():
# #     # Initialize MediaPipe FaceMesh and Drawing utilities
# #     mp_face_mesh = mp.solutions.face_mesh
# #     face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# #     mp_drawing = mp.solutions.drawing_utils
# #     drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0, 128))
# #
# #     # Load YOLOv5 model for cell phone detection
# #     yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
# #
# #     # Load MobileNetSSD model for person detection
# #     prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
# #     model_path = 'models/MobileNetSSD_deploy.caffemodel'
# #     net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# #     min_confidence = 0.5
# #     classes = {15: "person"}
# #     np.random.seed(543210)
# #     colors = np.random.uniform(0, 255, size=(max(classes.keys()) + 1, 3))
# #
# #     # Parameters for mouth opening detection
# #     threshold = 20  # Adjust this threshold based on your use case
# #     cheating_duration_threshold = 2.0  # Time in seconds for considering oral cheating
# #     cheating_frequency_threshold = 3  # Number of times mouth opened within a period to consider cheating
# #     cheating_period = 10  # Period in seconds to count the number of mouth openings
# #
# #     # Variables to track mouth opening
# #     mouth_open_start_time = None
# #     mouth_open_times = []
# #
# #     # Start video capture
# #     cap = cv2.VideoCapture(0)
# #
# #     while cap.isOpened():
# #         success, image = cap.read()
# #         if not success:
# #             break
# #
# #         # Flip the image horizontally for a later selfie-view display
# #         image = cv2.flip(image, 1)
# #
# #         # Preprocess the image
# #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         image.flags.writeable = False
# #         results = face_mesh.process(image_rgb)
# #         image.flags.writeable = True
# #         image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
# #
# #         img_h, img_w, img_c = image.shape
# #         face_3d, face_2d = [], []
# #
# #         if results.multi_face_landmarks:
# #             for face_landmarks in results.multi_face_landmarks:
# #                 landmarks_px = []
# #                 for idx, lm in enumerate(face_landmarks.landmark):
# #                     x, y = int(lm.x * img_w), int(lm.y * img_h)
# #                     if idx in [33, 263, 1, 61, 291, 199]:
# #                         if idx == 1:
# #                             nose_2d = (x, y)
# #                             nose_3d = (x, y, lm.z * 3000)
# #
# #                         face_2d.append([x, y])
# #                         face_3d.append([x, y, lm.z])
# #
# #                     landmarks_px.append((x, y))
# #
# #                 face_2d = np.array(face_2d, dtype=np.float64)
# #                 face_3d = np.array(face_3d, dtype=np.float64)
# #
# #                 # Camera matrix and distortion coefficients
# #                 focal_length = 1 * img_w
# #                 cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
# #                 dist_matrix = np.zeros((4, 1), dtype=np.float64)
# #
# #                 # Solve PnP for head pose estimation
# #                 success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
# #                 rmat, _ = cv2.Rodrigues(rot_vec)
# #                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
# #
# #                 x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
# #
# #                 if y < -10:
# #                     text = "Looking Left"
# #                 elif y > 10:
# #                     text = "Looking Right"
# #                 elif x < -10:
# #                     text = "Looking Down"
# #                 elif x > 10:
# #                     text = "Looking Up"
# #                 else:
# #                     text = "Forward"
# #
# #                 # Project nose point for visualization
# #                 nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
# #                 p1 = (int(nose_2d[0]), int(nose_2d[1]))
# #                 p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
# #
# #                 # Draw the line with transparency
# #                 overlay = image.copy()
# #                 cv2.line(overlay, p1, p2, (255, 0, 0), 3)
# #                 cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
# #
# #                 # Display text information
# #                 cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
# #                 cv2.putText(image, f"x: {x:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# #                 cv2.putText(image, f"y: {y:.2f}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# #                 cv2.putText(image, f"z: {z:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# #
# #                 # Draw face landmarks
# #                 for lm in face_landmarks.landmark:
# #                     x, y = int(lm.x * img_w), int(lm.y * img_h)
# #                     cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Small circle (pixel)
# #
# #                 # Mouth opening detection
# #                 mouth_top_px = landmarks_px[13]  # Upper lip top
# #                 mouth_bottom_px = landmarks_px[14]  # Lower lip bottom
# #                 distance = np.linalg.norm(np.array(mouth_top_px) - np.array(mouth_bottom_px))
# #                 current_time = time.time()
# #
# #                 if distance > threshold:
# #                     if mouth_open_start_time is None:
# #                         mouth_open_start_time = current_time
# #                     else:
# #                         duration = current_time - mouth_open_start_time
# #                         if duration > cheating_duration_threshold:
# #                             cv2.putText(image, 'Oral Cheating Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
# #                                         (0, 0, 255), 2, cv2.LINE_AA)
# #                 else:
# #                     if mouth_open_start_time is not None:
# #                         duration = current_time - mouth_open_start_time
# #                         if duration > cheating_duration_threshold:
# #                             cv2.putText(image, 'Oral Cheating Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
# #                                         (0, 0, 255), 2, cv2.LINE_AA)
# #                         mouth_open_times.append(current_time)
# #                         mouth_open_start_time = None
# #
# #                 mouth_open_times = [t for t in mouth_open_times if current_time - t <= cheating_period]
# #
# #                 if len(mouth_open_times) > cheating_frequency_threshold:
# #                     cv2.putText(image, 'Frequent Oral Cheating Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
# #                                 (0, 0, 255), 2, cv2.LINE_AA)
# #
# #         # YOLOv5 cell phone detection
# #         yolo_results = yolo_model(image)
# #         cell_phone_detected = False
# #         for box in yolo_results.xyxy[0]:
# #             x1, y1, x2, y2, confidence, cls = box
# #             if int(cls) == 67:  # Class ID for cell phones
# #                 cell_phone_detected = True
# #                 cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
# #                 cv2.putText(image, f'Cell Phone Detected', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
# #                             (0, 255, 0), 2)
# #
# #         # Person detection
# #         blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
# #         net.setInput(blob)
# #         detections = net.forward()
# #
# #         person_detected = False
# #         for i in np.arange(0, detections.shape[2]):
# #             confidence = detections[0, 0, i, 2]
# #             if confidence > min_confidence:
# #                 idx = int(detections[0, 0, i, 1])
# #                 if idx in classes.keys():
# #                     person_detected = True
# #                     box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
# #                     (startX, startY, endX, endY) = box.astype("int")
# #                     label = f"{classes[idx]}: {confidence:.2f}%"
# #                     cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 2)
# #                     y = startY - 15 if startY - 15 > 15 else startY + 15
# #                     cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
# #
# #         # Display number of persons detected, mobile phone status, and mouth status
# #         cv2.putText(image, f'Persons Detected: {person_detected}', (10, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
# #                     (255, 0, 0), 2)
# #         cv2.putText(image, f'Mobile Phone Detected: {"Yes" if cell_phone_detected else "No"}',
# #                     (10, img_h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
# #         cv2.putText(image, f'Mouth Open: {"Yes" if mouth_open_start_time is not None else "No"}',
# #                     (10, img_h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
# #
# #         # Show the output
# #         cv2.imshow('Exam Proctoring System', image)
# #
# #         if cv2.waitKey(5) & 0xFF == 27:
# #             break
# #
# #     cap.release()
# #     cv2.destroyAllWindows()
# #
# # # Run the proctoring system
# # run_proctoring_system()
# import cv2
# import numpy as np
# from detect_head_movement import detect_head_pose, detect_head_movement
# from detectPerson import detect_objects
# from detectPhone import detect_cell_phones
# from mouth_detection_movement import detect_mouth_opening
#
# # Initialize video capture
# cap = cv2.VideoCapture(0)
#
# # Variables for head pose and mouth detection
# head_direction = "Unknown"
# mouth_open_start_time = None
# mouth_open_times = []
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success or frame is None:
#         print("Failed to capture frame or frame is None")
#         continue
#
#     # Head Pose Estimation
#     try:
#         results_head_pose = detect_head_pose(frame)
#     except ValueError as e:
#         print(e)
#         continue
#
#     if results_head_pose and results_head_pose.multi_face_landmarks:
#         # Process head pose results
#         landmarks = results_head_pose.multi_face_landmarks[0].landmark
#         head_direction = detect_head_movement(landmarks)
#
#     # Object Detection
#     detections, (h, w) = detect_objects(frame)
#     person_count = 0
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.2:
#             class_index = int(detections[0, 0, i, 1])
#             if class_index == 15:  # Person class index
#                 person_count += 1
#             if class_index in [15, 25]:  # Checking for person and phone
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 color = (0, 255, 0) if class_index == 15 else (0, 0, 255)
#                 label = "Person" if class_index == 15 else "Phone"
#                 cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
#                 cv2.putText(frame, f"{label}: {confidence:.2f}",
#                             (startX, startY - 15 if startY > 30 else startY + 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     # Cell Phone Detection
#     results_cell_phone = detect_cell_phones(frame)
#     for box in results_cell_phone.xyxy[0]:
#         x1, y1, x2, y2, confidence, cls = box
#         if int(cls) == 67:  # Class ID for cell phones
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f'Cell Phone: {confidence:.2f}', (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#
#     # Mouth Opening Detection
#     if results_head_pose and results_head_pose.multi_face_landmarks:
#         detect_mouth_opening(results_head_pose, frame, mouth_open_start_time, mouth_open_times)
#
#     # Display Summary Information
#     summary_frame = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
#     cv2.putText(summary_frame, f'Number of Persons: {person_count}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(summary_frame, f'Head Direction: {head_direction}', (10, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(summary_frame, 'Mouth Movement: Checking...', (10, 110),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     if mouth_open_start_time:
#         cv2.putText(summary_frame, 'Mouth Opening Detected', (10, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     elif len(mouth_open_times) > 3:  # Example condition for frequent movement
#         cv2.putText(summary_frame, 'Frequent Mouth Movement Detected', (10, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     else:
#         cv2.putText(summary_frame, 'No Mouth Movement Detected', (10, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#     # Display frames
#     cv2.imshow('Integrated Detection System', frame)
#     cv2.imshow('Summary Information', summary_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
#
import cv2
import time
from detect_head_movement import initialize_face_mesh as init_head_pose, process_frame as process_head_pose
from detectPhone import load_model as load_yolo, perform_detection as detect_cell_phone, process_results as process_cell_phone
from mouth_detection_movement import initialize_face_mesh as init_mouth_detection, process_image as process_mouth_image, get_landmarks, detect_mouth_opening, draw_results as draw_mouth_results
from detectPerson import load_model as load_ssd, initialize_classes_and_colors, preprocess_frame, detect_objects, process_detections

# def main():
#     # Initialize all models and required objects
#     face_mesh_head_pose = init_head_pose()
#     yolo_model = load_yolo()
#     face_mesh_mouth = init_mouth_detection()
#     ssd_net = load_ssd('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')
#     classes, colors = initialize_classes_and_colors()
#
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
#
#     # Parameters for mouth opening detection
#     threshold = 20
#     cheating_duration_threshold = 2.0
#     cheating_frequency_threshold = 3
#     cheating_period = 10
#     mouth_open_start_time = None
#     mouth_open_times = []
#
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print("Failed to grab frame")
#             break
#
#         # Head Pose Estimation
#         head_pose_frame = process_head_pose(frame.copy(), face_mesh_head_pose)
#
#         # Cell Phone Detection
#         cell_phone_results = detect_cell_phone(frame, yolo_model)
#         cell_phone_frame, _ = process_cell_phone(frame.copy(), cell_phone_results)
#
#         # Mouth Opening Detection
#         mouth_results, mouth_frame = process_mouth_image(frame.copy(), face_mesh_mouth)
#         if mouth_results.multi_face_landmarks:
#             for face_landmarks in mouth_results.multi_face_landmarks:
#                 landmarks_px = get_landmarks(face_landmarks, mouth_frame.shape)
#                 current_time = time.time()
#                 cheating_detected, mouth_open_start_time, mouth_open_times = detect_mouth_opening(
#                     landmarks_px, threshold, current_time, mouth_open_start_time, mouth_open_times,
#                     cheating_duration_threshold, cheating_period
#                 )
#                 mouth_frame = draw_mouth_results(mouth_frame, cheating_detected, mouth_open_times, cheating_frequency_threshold)
#
#         # Webcam Object Detection
#         blob, h, w = preprocess_frame(frame)
#         detected_objects = detect_objects(ssd_net, blob)
#         webcam_frame = process_detections(frame.copy(), detected_objects, classes, colors, 0.2, h, w)
#
#         # Combine all frames
#         top_row = cv2.hconcat([head_pose_frame, cell_phone_frame])
#         bottom_row = cv2.hconcat([mouth_frame, webcam_frame])
#         combined_frame = cv2.vconcat([top_row, bottom_row])
#
#         # Resize if the combined frame is too large
#         scale_percent = 50  # percent of original size
#         width = int(combined_frame.shape[1] * scale_percent / 100)
#         height = int(combined_frame.shape[0] * scale_percent / 100)
#         dim = (width, height)
#         resized_frame = cv2.resize(combined_frame, dim, interpolation=cv2.INTER_AREA)
#
#         # Display the combined frame
#         cv2.imshow('Combined Detection Systems', resized_frame)
#
#         if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
def main():
    # Initialize all models and required objects
    face_mesh_head_pose = init_head_pose()
    yolo_model = load_yolo()
    face_mesh_mouth = init_mouth_detection()
    ssd_net = load_ssd('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')
    classes, colors = initialize_classes_and_colors()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Parameters for mouth opening detection
    threshold = 20
    cheating_duration_threshold = 2.0
    cheating_frequency_threshold = 3
    cheating_period = 10
    mouth_open_start_time = None
    mouth_open_times = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Head Pose Estimation
        frame = process_head_pose(frame, face_mesh_head_pose)

        # Cell Phone Detection
        cell_phone_results = detect_cell_phone(frame, yolo_model)
        frame = process_cell_phone(frame, cell_phone_results)

        # Mouth Opening Detection
        if isinstance(frame, tuple):  # Check if frame is a tuple
            frame = frame[0]  # If it's a tuple, use the first element
        mouth_results, _ = process_mouth_image(frame, face_mesh_mouth)
        if mouth_results.multi_face_landmarks:
            for face_landmarks in mouth_results.multi_face_landmarks:
                landmarks_px = get_landmarks(face_landmarks, frame.shape)
                current_time = time.time()
                cheating_detected, mouth_open_start_time, mouth_open_times = detect_mouth_opening(
                    landmarks_px, threshold, current_time, mouth_open_start_time, mouth_open_times,
                    cheating_duration_threshold, cheating_period
                )
                frame = draw_mouth_results(frame, cheating_detected, mouth_open_times, cheating_frequency_threshold)

        # Webcam Object Detection
        blob, h, w = preprocess_frame(frame)
        detected_objects = detect_objects(ssd_net, blob)
        frame = process_detections(frame, detected_objects, classes, colors, 0.2, h, w)

        # Display the combined frame
        cv2.imshow('Combined Detection Systems', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()