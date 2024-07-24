import cv2
import numpy as np

# Load the video capture device (in this case, the virtual camera)
cap = cv2.VideoCapture(1)  # Replace 1 with the index of your virtual camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of HSV values for a mobile phone
    lower_phone = np.array([0, 0, 100])
    upper_phone = np.array([180, 255, 255])

    # Threshold the HSV image to get only the mobile phone
    mask = cv2.inRange(hsv, lower_phone, upper_phone)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and draw a bounding rectangle around the mobile phone
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this value to filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Mobile Phone Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()