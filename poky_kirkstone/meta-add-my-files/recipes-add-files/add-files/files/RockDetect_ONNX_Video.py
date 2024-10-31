import cv2
import numpy as np

# Load the ONNX model using OpenCV DNN module
model_path = "best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

# Open the video file
video_path = "VideoRocasPrueba.mp4"
pipeline = f"filesrc location={video_path} ! queue ! decodebin ! queue ! videoconvert ! video/x-raw, format=BGR ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# NMS parameters
confidence_threshold = 0.5
nms_threshold = 0.4

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    #frame = cv2.resize(frame,(320,240))
    
    # Break the loop if no frame is retrieved (end of video)
    if not ret:
        break

    # Get the original frame size
    original_height, original_width = frame.shape[:2]

    # Preprocess the frame (resize to 640x640 for YOLOv8)
    input_size = 640
    input_frame = cv2.resize(frame, (input_size, input_size))
    input_blob = cv2.dnn.blobFromImage(input_frame, scalefactor=1/255.0, size=(input_size, input_size), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(input_blob)

    # Run forward pass to get the predictions
    output = net.forward()

    # Output shape is (1, 5, 8400)
    # Reshape the output if needed, in case it is not already shaped as expected
    output = output.reshape((5, 8400))

    # Prepare lists for boxes and confidences
    boxes = []
    confidences = []

    # Parse the output
    for i in range(output.shape[1]):  # Loop over the 8400 predictions
        x_center, y_center, width_box, height_box, confidence = output[:, i]

        if confidence > confidence_threshold:  # Threshold for detection (adjust as needed)
            # Rescale box coordinates back to original frame size
            x_center *= original_width
            y_center *= original_height
            width_box *= original_width
            height_box *= original_height
            
            x = int(x_center - width_box / 2)
            y = int(y_center - height_box / 2)

            # Store the boxes and confidences
            boxes.append([x, y, int(width_box), int(height_box)])
            confidences.append(float(confidence))

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw the bounding boxes on the original frame
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Conf: {confidences[i]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("Detections", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
