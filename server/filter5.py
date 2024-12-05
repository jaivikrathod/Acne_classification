import cv2
import numpy as np

# Function to detect acne marks in the image
def detect_acne_marks(image):
    # Convert the image to HSV color space for color-based thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define multiple HSV ranges for red and pinkish tones
    hsv_ranges = [
        (np.array([0, 50, 50]), np.array([20, 255, 255])),    # Red tones
        (np.array([140, 50, 50]), np.array([180, 255, 255]))  # Pink tones
    ]

    # Initialize an empty mask
    final_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Apply each HSV range and combine masks
    for lower_range, upper_range in hsv_ranges:
        mask = cv2.inRange(hsv, lower_range, upper_range)
        final_mask = cv2.bitwise_or(final_mask, mask)

    # Find contours of potential acne marks in the combined mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected acne areas on the image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # if((x+30) > (x+w)):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangles

    return image

# Function to apply filters to enhance the image
def apply_filters(image, brightness=-100, contrast=250, saturation=80, sharpness=100):
    from PIL import Image, ImageEnhance
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1 + brightness / 255.0)

    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1 + contrast / 255.0)

    # Convert back to OpenCV format
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Saturation adjustment
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Sharpness adjustment using a kernel
    kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 25, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return image

# Main function to load image, apply filters, and mark acne
def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at '{image_path}'")
        return

    # Apply filters
    filtered_image = apply_filters(image)
    # filtered_image = image

    # Detect acne marks in the entire image
    processed_image = detect_acne_marks(filtered_image)

    # Display the final image with detected acne marks
    cv2.imshow('Detected Acne Marks', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function with the path to your image
# main(image_path)
# main('acne4.png')
