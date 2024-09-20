# import cv2
# import torch
#
# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
#
# # Load the video or webcam feed
# cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # Perform detection
#     results = model(frame)
#
#     # Initialize a flag for cell phone detection
#     cell_phone_detected = False
#
#     # Draw bounding boxes on the frame and add text
#     for box in results.xyxy[0]:  # results.xyxy[0] returns bounding boxes in [x1, y1, x2, y2, confidence, class]
#         x1, y1, x2, y2, confidence, cls = box
#         if int(cls) == 67:  # '67' is the class ID for cell phones in COCO dataset
#             cell_phone_detected = True
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f'Cell Phone: {confidence:.2f}', (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#
#     # Display the detection status text on the frame
#     if cell_phone_detected:
#         cv2.putText(frame, 'Cell Phone Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#     # Display the frame with detections
#     cv2.imshow('Cell Phone Detection', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch


def load_model(model_path='yolov5s.pt'):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)


def perform_detection(frame, model):
    return model(frame)


def process_results(frame, results):
    cell_phone_detected = False
    for box in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = box
        if int(cls) == 67:  # '67' is the class ID for cell phones in COCO dataset
            cell_phone_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Cell Phone: {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    if cell_phone_detected:
        cv2.putText(frame, 'Cell Phone Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, cell_phone_detected


def initialize_capture(source=0):
    return cv2.VideoCapture(source)


def display_frame(frame, window_name='Cell Phone Detection'):
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF == ord('q')


def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()