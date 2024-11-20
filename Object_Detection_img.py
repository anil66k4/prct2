import cv2
import numpy as np

# Paths to the custom YOLOv3 configuration, weights, and classes files
cfg_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/yolov3.cfg'
weights_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/yolov3 .weights'
names_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/coco.names'

# Load the custom class names
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Load the network
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)


# Load the image you want to test
image_path = 'C:/Users/anilr/Downloads/YOLOv3-Custom-Object-Detection-main/test.jpg'  # Replace with your image path
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

# Preprocess the image: Create a 4D blob from the input image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get YOLO output layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform forward pass and get output
layer_outputs = net.forward(layer_names)

# Initialize lists to hold detected bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Set minimum confidence threshold
conf_threshold = 0.03  # Lowering the confidence threshold for more detections
nms_threshold = 0.4

# Iterate over each of the layer outputs
for output in layer_outputs:
    for detection in output:
        # Extract the class scores and determine the predicted class ID
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Only proceed if confidence is above the threshold
        if confidence > conf_threshold:
            # Scale the bounding box coordinates back to the size of the image
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # Calculate the top-left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression to suppress weak, overlapping boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Check if any detections exist
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # Draw the bounding box and label
        color = (255, 255, 0)  # Green for detections
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.4f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
else:
    print("No objects detected")

# Show the output image with detections
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Number of boxes detected:", len(boxes))
print("Confidences:", confidences)
