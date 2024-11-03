# import cv2
# import mediapipe as mp
# import numpy as np
# import time
#
# # Initialize MediaPipe FaceMesh and Drawing utilities
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils
#
# # Define drawing specifications with reduced thickness and circle radius
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0, 128))
#
# # Start video capture
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         break
#
#     start = time.time()
#
#     # Preprocess the image
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = face_mesh.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     img_h, img_w, img_c = image.shape
#     face_3d, face_2d = [], []
#
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for idx, lm in enumerate(face_landmarks.landmark):
#                 if idx in [33, 263, 1, 61, 291, 199]:  # Reduce to important landmarks
#                     x, y = int(lm.x * img_w), int(lm.y * img_h)
#                     if idx == 1:
#                         nose_2d = (x, y)
#                         nose_3d = (x, y, lm.z * 3000)
#
#                     face_2d.append([x, y])
#                     face_3d.append([x, y, lm.z])
#
#             face_2d = np.array(face_2d, dtype=np.float64)
#             face_3d = np.array(face_3d, dtype=np.float64)
#
#             # Camera matrix and distortion coefficients
#             focal_length = 1 * img_w
#             cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
#             dist_matrix = np.zeros((4, 1), dtype=np.float64)
#
#             # Solve PnP
#             success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#             rmat, _ = cv2.Rodrigues(rot_vec)
#             angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#
#             x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
#
#             if y < -10:
#                 text = "Looking Left"
#             elif y > 10:
#                 text = "Looking Right"
#             elif x < -10:
#                 text = "Looking Down"
#             elif x > 10:
#                 text = "Looking Up"
#             else:
#                 text = "Forward"
#
#             # Project nose point for visualization
#             nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
#             p1 = (int(nose_2d[0]), int(nose_2d[1]))
#             p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
#
#             # Draw the line with transparency
#             overlay = image.copy()
#             cv2.line(overlay, p1, p2, (255, 0, 0), 3)
#             cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
#
#             # Display text information
#             cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
#             cv2.putText(image, f"x: {x:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(image, f"y: {y:.2f}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(image, f"z: {z:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#             # Draw face landmarks as pixels
#             for lm in face_landmarks.landmark:
#                 x, y = int(lm.x * img_w), int(lm.y * img_h)
#                 cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Small circle (pixel)
#
#     cv2.imshow('Head Pose Estimation', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# another

# import cv2
# import mediapipe as mp
# import numpy as np
# import time
#
#
# def detect_head_movement():
#     # Initialize MediaPipe FaceMesh and Drawing utilities
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils
#
#     # Define drawing specifications with reduced thickness and circle radius
#     drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0, 128))
#
#     # Start video capture
#     cap = cv2.VideoCapture(0)
#
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             break
#
#         start = time.time()
#
#         # Preprocess the image
#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = face_mesh.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         img_h, img_w, img_c = image.shape
#         face_3d, face_2d = [], []
#
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 for idx, lm in enumerate(face_landmarks.landmark):
#                     if idx in [33, 263, 1, 61, 291, 199]:  # Reduce to important landmarks
#                         x, y = int(lm.x * img_w), int(lm.y * img_h)
#                         if idx == 1:
#                             nose_2d = (x, y)
#                             nose_3d = (x, y, lm.z * 3000)
#
#                         face_2d.append([x, y])
#                         face_3d.append([x, y, lm.z])
#
#                 face_2d = np.array(face_2d, dtype=np.float64)
#                 face_3d = np.array(face_3d, dtype=np.float64)
#
#                 # Camera matrix and distortion coefficients
#                 focal_length = 1 * img_w
#                 cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
#                 dist_matrix = np.zeros((4, 1), dtype=np.float64)
#
#                 # Solve PnP
#                 success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#                 rmat, _ = cv2.Rodrigues(rot_vec)
#                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#
#                 x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
#
#                 if y < -10:
#                     text = "Looking Left"
#                 elif y > 10:
#                     text = "Looking Right"
#                 elif x < -10:
#                     text = "Looking Down"
#                 elif x > 10:
#                     text = "Looking Up"
#                 else:
#                     text = "Forward"
#
#                 # Project nose point for visualization
#                 nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
#                 p1 = (int(nose_2d[0]), int(nose_2d[1]))
#                 p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
#
#                 # Draw the line with transparency
#                 overlay = image.copy()
#                 cv2.line(overlay, p1, p2, (255, 0, 0), 3)
#                 cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
#
#                 # Display text information
#                 cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
#                 cv2.putText(image, f"x: {x:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 cv2.putText(image, f"y: {y:.2f}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 cv2.putText(image, f"z: {z:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#                 # Draw face landmarks as pixels
#                 for lm in face_landmarks.landmark:
#                     x, y = int(lm.x * img_w), int(lm.y * img_h)
#                     cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Small circle (pixel)
#
#         cv2.imshow('Head Pose Estimation', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def initialize_face_mesh():
    return mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process_image(image, face_mesh):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def get_face_landmarks(results, img_w, img_h):
    face_3d, face_2d = [], []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                if idx == 1:
                    nose_2d = (x, y)
                    nose_3d = (x, y, lm.z * 3000)

    return np.array(face_3d, dtype=np.float64), np.array(face_2d, dtype=np.float64), nose_2d, nose_3d


def calculate_pose(face_3d, face_2d, img_w, img_h):
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                           [0, focal_length, img_h / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return rot_vec, trans_vec, cam_matrix, dist_matrix, angles


def determine_head_pose(angles):
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
    return text, x, y, z


def draw_pose_info(image, nose_2d, nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix, text, x, y, z):
    nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    cv2.line(image, p1, p2, (255, 0, 0), 3)

    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image, f"x: {x:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, f"y: {y:.2f}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, f"z: {z:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image


def process_frame(image, face_mesh):
    image, results = process_image(image, face_mesh)
    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        face_3d, face_2d, nose_2d, nose_3d = get_face_landmarks(results, img_w, img_h)
        rot_vec, trans_vec, cam_matrix, dist_matrix, angles = calculate_pose(face_3d, face_2d, img_w, img_h)
        text, x, y, z = determine_head_pose(angles)
        image = draw_pose_info(image, nose_2d, nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix, text, x, y, z)

    return image

