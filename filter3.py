import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Function to apply brightness, contrast, and saturation adjustments
def apply_filters(image, brightness=-100, contrast=250, shadows=0, saturation=80, sharpness=100):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1 + brightness / 255.0)

    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1 + contrast / 255.0)

    # Convert back to OpenCV format
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Shadows adjustment
    shadows_value = abs(shadows)
    image = np.where(image < shadows_value, image * (1 - shadows / 255.0), image).astype(np.uint8)

    # Saturation adjustment
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Sharpness adjustment using a kernel
    kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 25, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return image

# Function to detect acne marks in a specific region of a face
def detect_acne_marks(face_roi):
    # Convert face ROI to HSV color space for color-based thresholding
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    # Define the HSV range for pinkish shades
    
    # lower_range = np.array([140, 50, 50])
    # upper_range = np.array([186, 243, 243])
    
    lower_range = np.array([255,96,0])
    upper_range = np.array([168,17,2])


    # Create a mask to isolate pinkish areas
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Find contours of potential acne marks in the pinkish areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected pinkish areas on the face ROI
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangles

    return face_roi

# Main function to load image, apply filters, detect faces, and mark acne in a specific region
def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at '{image_path}'")
        return

    # Apply filters
    filtered_image = apply_filters(image)

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade model for face detection
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces

    faces = gray_image.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in gray_image:
        # Define a specific rectangular region within the face
        roi_x_start, roi_x_end = x + 70, x + w - 70
        roi_y_start, roi_y_end = y, y + h
        face_roi = filtered_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Detect acne marks within the defined rectangle
        processed_face_roi = detect_acne_marks(face_roi)

        # Place the processed region back into the main image
        filtered_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = processed_face_roi

        # Draw the rectangle outline on the main image
        cv2.rectangle(filtered_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

    # Display the final image with detected faces and acne marks
    cv2.imshow('Detected Faces with Acne Marks', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the number of faces found
    # print(f"Found {len(faces)} face(s) in the image.")

# Run the function with the path to your image
main('acne9.jpg')
