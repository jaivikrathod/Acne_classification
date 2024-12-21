import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Read an image
image = cv2.imread('images.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face mesh detection
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape
        
        # Collect landmarks for eyes and lips
        eye_landmarks = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]  # Eye region
        eyebrow_landmarks = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]  # Eyebrow region
        lip_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]  # Lip region
        
        def draw_white_rectangle(landmarks_indices):
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for idx in landmarks_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        
        # Remove eyes with eyebrows
        draw_white_rectangle(eye_landmarks + eyebrow_landmarks)
        
        # Remove lips
        draw_white_rectangle(lip_landmarks)

# Display the result
cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
