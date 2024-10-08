import numpy as np
import cv2

# Paths to the model files
image_path = 'lena.png'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

# Define class labels and colors
classes = {15: "person", 25: "phone"}
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(max(classes.keys()) + 1, 3))  # Ensure enough colors

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load and process the image
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)
detected_objects = net.forward()

# Iterate over detections and filter by class
for i in range(detected_objects.shape[2]):
    confidence = detected_objects[0, 0, i, 2]
    if confidence > min_confidence:
        class_index = int(detected_objects[0, 0, i, 1])

        # Check if class_index is in the defined classes
        if class_index in classes:
            # Convert bounding box coordinates
            box = detected_objects[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Prepare text and color
            prediction_text = f"{classes[class_index]}: {confidence:.2f}"
            color = tuple(map(int, colors[class_index]))

            # Draw bounding box and label
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            cv2.putText(image, prediction_text,
                        (startX, startY - 15 if startY > 30 else startY + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            print(f"Class index {class_index} not found in predefined classes.")

# Display the output
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()