import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Read an image
image = cv2.imread('test2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face mesh detection
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape

        # Collect landmarks for the upper face region (approximation for the hairline)
        upper_face_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]  # Rough upper face landmarks
        
        def get_region_mask(landmarks_indices):
            """Create a mask for the given landmarks region."""
            mask = np.zeros((h, w), dtype=np.uint8)
            points = []
            for idx in landmarks_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                points.append((x, y))
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            return mask

        # Create mask for the hair region
        hair_mask = get_region_mask(upper_face_landmarks)

        # Extract only the hair pixels (black and brown shades)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color range for black and brown hair in HSV
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])  # Black range
        brown_lower = np.array([10, 100, 20])
        brown_upper = np.array([20, 255, 200])  # Brown range

        # Create masks for black and brown shades
        black_mask = cv2.inRange(hsv_image, black_lower, black_upper)
        brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)

        # Combine black and brown masks and apply to hair region
        hair_color_mask = cv2.bitwise_or(black_mask, brown_mask)
        hair_masked = cv2.bitwise_and(hair_color_mask, hair_color_mask, mask=hair_mask)

        # Remove hair by replacing the region with white
        image[np.where(hair_masked > 0)] = [255, 255, 255]

# Display the result
cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
