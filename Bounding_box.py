import cv2
import numpy as np
import time
#import lane_lines
# Load YOLO model and COCO labels
model_path = '/home/nishadh/Models/yolov3-tiny.weights'  # Use YOLOv3-tiny for better performance
config_path = '/home/nishadh/Models/yolov3-tiny.cfg'     # YOLOv3-tiny config file
labels_path = '/home/nishadh/Models/coco-labels.txt'     # COCO labels file

# Load the network
net = cv2.dnn.readNet(model_path, config_path)
with open(labels_path, 'r') as f:
    labels = f.read().strip().split("\n")

# Set preferable backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize camera
camera = cv2.VideoCapture(0)

# Set a lower resolution for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Set confidence and NMS thresholds
conf_threshold = 0.3
nms_threshold = 0.2

# Skip frames to increase FPS
frame_count = 0
skip_frames = 1  # Process every 3rd frame

def get_color_based_on_distance(distance):
    if distance < 4:  # Distance less than 4 meters
        return (0, 0, 255)  # Red
    elif distance > 5:  # Distance greater than 5 meters
        return (0, 255, 0)  # Green
    else:
        return (0, 255, 255)  # Yellow (between 4 and 5 meters)

while True:
    frame_count += 1
    ret, frame = camera.read()
    if not ret or frame_count % skip_frames != 0:
        continue
    
    height, width = frame.shape[:2]

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Parse YOLO outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - w / 2)
                y = int(centerY - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes with distance-based color coding
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = str(labels[class_ids[i]])
            confidence = confidences[i]

            # Estimate distance (simplified placeholder formula)
            distance = (1.0 - confidence) * 10  # Adjust as needed
            color = get_color_based_on_distance(distance)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()

