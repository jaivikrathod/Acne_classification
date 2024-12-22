from mtcnn import MTCNN
import cv2

# Load the MTCNN detector
detector = MTCNN()

# Read an image
image = cv2.imread('acne_side1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection
faces = detector.detect_faces(image_rgb)

# Draw bounding boxes
for face in faces:
    x, y, width, height = face['box']
    confidence = face['confidence']
    print(x,y,width,height) 
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(image, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(len(faces))
# Display the result
cv2.imshow('MTCNN Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()