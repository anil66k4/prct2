import cv2
import numpy as np

# Paths to the custom YOLOv3 configuration and weights files
cfg_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/yolov3.cfg'
weights_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/yolov3 .weights'
names_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/coco.names'

# Load the custom class names
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

print(f"Classes: {classes}")  # Debug: Check if class names are loaded correctly

# Load the network
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Use GPU for inference (if available)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Set up the video capture
video_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/video.mp4'
cap = cv2.VideoCapture(video_path)

# Function to process detections
def detect_objects(frame):
    (H, W) = frame.shape[:2]
    
    # Create a 4D blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO layer names
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Perform forward pass and get output
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions
            if confidence > 0.3 and class_id == classes.index('tiger'):  # Adjust confidence threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                print(f"Detected: {classes[class_id]} with confidence {confidence}")  # Debugging line

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.5)

    return boxes, confidences, class_ids, idxs

# Loop over video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, confidences, class_ids, idxs = detect_objects(frame)

    # If there are detections, draw them on the frame
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw bounding box and label
            color = (0, 255, 0)  # Green for tiger
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
