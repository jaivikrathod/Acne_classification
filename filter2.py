import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Function to apply color correction to remove green tint
# def remove_green_tint(image):
#     # Reduce the green channel intensity
#     corrected_image = image.copy()
#     corrected_image[:, :, 1] = (corrected_image[:, :, 1] * 0.5).astype(np.uint8)
#     return corrected_image

# Function to apply brightness, contrast, and saturation adjustments
def apply_filters(image, brightness=-100, contrast=250, shadows=0, saturation=80, sharpness=100):
    # Convert to PIL Image for easier manipulation of brightness, contrast, and saturation
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1 + brightness / 255.0)

    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1 + contrast / 255.0)

    # Convert back to OpenCV format to apply remaining filters
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Shadows adjustment
    shadows_value = abs(shadows)
    image = np.where(image < shadows_value, image * (1 - shadows / 255.0), image).astype(np.uint8)

    # Saturation adjustment
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Sharpness adjustment using a kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9 + sharpness / 25, -1],
                       [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return image

# Function to detect acne marks on a face region of interest (ROI)
# def detect_acne_marks(face_roi):
    # Convert face ROI to grayscale for better thresholding
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to isolate potential acne marks
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours of potential acne marks
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected acne marks on the face ROI
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangles around acne marks

    return face_roi


def detect_acne_marks(face_roi):
    # Convert face ROI to HSV color space for color-based thresholding
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    # Define the HSV range for pinkish shades (adjust as needed)
    # Lower and upper bounds for dark pink, light pink, and peach shades
    lower_pink = np.array([140, 50, 50])    # Example values for lower bound
    upper_pink = np.array([180, 255, 255])  # Example values for upper bound

    # Create a mask to isolate pinkish areas
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Find contours of potential acne marks in the pinkish areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected pinkish areas on the face ROI
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangles around detected areas

    return face_roi

# Main function to load image, apply filters, color correction, detect faces, and mark acne
def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at '{image_path}'")
        return

    # Apply brightness, contrast, shadows, saturation, and sharpness adjustments
    filtered_image = apply_filters(image)
    # filtered_image = image

    # Apply color correction to remove green tint
    # corrected_image = remove_green_tint(filtered_image)

    # Convert the corrected image to grayscale for face detection
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the corrected image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.01, minNeighbors=5, minSize=(30, 30))

    # Process each detected face for acne marks
    for (x, y, w, h) in faces:
        face_roi = filtered_image[y:y+h, x:x+w]  # Region of interest (face)
        image_with_acne_marks = detect_acne_marks(face_roi)
        # Place the processed face ROI with acne marks back into the corrected image
        filtered_image[y:y+h, x:x+w] = image_with_acne_marks

    # Display the final image with detected faces and acne marks
    cv2.imshow('Detected Faces with Acne Marks', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the number of faces found
    print(f"Found {len(faces)} face(s) in the image.")

# Run the function with the path to your image
main('acne9.jpg')
# main('acne9.png')
# main('acne_side2.jpg')