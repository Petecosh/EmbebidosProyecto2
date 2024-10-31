import cv2
import numpy as np

# Load the ONNX model using OpenCV DNN module
model_path = "best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

# Load an image
image = cv2.imread("Rocas_Prueba.jpg")
original_height, original_width = image.shape[:2]

print("Original height =", original_height, "Original width =", original_width)

# Preprocess the image (resize to 640x640 for YOLOv8)
input_size = 640
input_image = cv2.resize(image, (input_size, input_size))
input_blob = cv2.dnn.blobFromImage(input_image, scalefactor=1/255.0, size=(input_size, input_size), swapRB=True, crop=False)

# Set the input to the model
net.setInput(input_blob)

# Run forward pass to get the predictions
output = net.forward()

print(output.shape)

# Output shape is (1, 5, 8400)
# Reshape the output if needed, in case it is not already shaped as expected
output = output.reshape((5, 8400))

# Prepare lists for boxes and confidences
boxes = []
confidences = []

# Parse the output
for i in range(output.shape[1]):  # Loop over the 8400 predictions
    x_center, y_center, width_box, height_box, confidence = output[:, i]

    if confidence > 0.5:  # Threshold for detection (adjust as needed)
        print(x_center, y_center, width_box, height_box, confidence)
        # Rescale box coordinates back to original image size
        x_center *= original_width
        y_center *= original_height
        width_box *= original_width
        height_box *= original_height
        
        x = int(x_center - width_box / 2)
        y = int(y_center - height_box / 2)

        # Store the boxes and confidences
        boxes.append([x, y, int(width_box), int(height_box)])
        confidences.append(float(confidence))

# Print out the number of detections
num_detections = len(boxes)
print(f"Number of detections found: {num_detections}")

# Check if any boxes were detected
if not boxes:
    print("No objects detected")
else:
    # Draw the bounding boxes on the original image
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'Conf: {confidences[i]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the image with detections
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()