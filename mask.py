import cv2
import mediapipe as mp
import numpy as np

# Load the image
image_path = 'insan.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect the pose
results = pose.process(image_rgb)

# Create an empty mask
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# If a pose is detected
if results.pose_landmarks:
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Convert pose landmarks to numpy array and scale to image size
    landmarks = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in results.pose_landmarks.landmark])

    # Create a convex hull around the landmarks to form a rough mask of the human figure
    hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, hull, 255)

    # Draw each pair of connected landmarks to enhance the mask
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(mask, tuple(landmarks[start_idx]), tuple(landmarks[end_idx]), 255, thickness=10)

    # Optional: Additional contours around the body for better coverage
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Refine the mask with morphological operations
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Final mask adjustments for sharp edges and accuracy
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Save the mask as an output file
output_path = 'output.png'
cv2.imwrite(output_path, mask)

# Cleanup
pose.close()
