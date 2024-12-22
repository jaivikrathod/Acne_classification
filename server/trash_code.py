import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp

detector = MTCNN()



detected_face_x=0
detected_face_y=0
detected_face_width=0
detected_face_height=0

# Function to detect acne marks in the image
def detect_acne_marks(face_detected_image):

    removed_eye_pic = detect_eyes_lips(face_detected_image)
    image = apply_filters(removed_eye_pic)
    # image = removed_eye_pic
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
    counter =0
    # Draw rectangles around detected acne areas on the image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if(((detected_face_x<x) and ((detected_face_x+detected_face_width)>(x+w)) and (detected_face_y<y) and (detected_face_y+detected_face_height)>(y+h))):
         if((x+w) > (x+3) and(x+w) < (x+20) and (y+h) < (y+20) ):
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangles
          counter+=1
    
    cv2.imshow('Detected Faces with Acne Marks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return counter

# Function to apply filters to enhance the image
# def apply_filters(image, brightness=-100, contrast=250, saturation=80, sharpness=100):
#     from PIL import Image, ImageEnhance
#     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Brightness adjustment
#     enhancer = ImageEnhance.Brightness(pil_image)
#     pil_image = enhancer.enhance(1 + brightness / 255.0)

#     # Contrast adjustment
#     enhancer = ImageEnhance.Contrast(pil_image)
#     pil_image = enhancer.enhance(1 + contrast / 255.0)

#     # Convert back to OpenCV format
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

#     # Saturation adjustment
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation)
#     image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

#     # Sharpness adjustment using a kernel
#     kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 25, -1], [-1, -1, -1]])
#     image = cv2.filter2D(image, -1, kernel)

#     return image





def apply_filters(image, brightness=-100, contrast=200, saturation=100, sharpness=100, redness_boost=5):
    from PIL import Image, ImageEnhance
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # # Brightness adjustment
    # enhancer = ImageEnhance.Brightness(pil_image)
    # pil_image = enhancer.enhance(1 + brightness / 255.0)

    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1 + contrast / 255.0)

    # Convert back to OpenCV format
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Saturation adjustment
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation)

    # Boost redness by increasing saturation and value for red tones
    lower_red_1 = np.array([0, 50, 50])  # Lower range for red
    upper_red_1 = np.array([20, 255, 255])
    lower_red_2 = np.array([160, 50, 50])  # Upper range for red
    upper_red_2 = np.array([200, 255, 255])

    # Create masks for red tones
    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply redness boost to the masked areas
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], redness_boost, mask=red_mask)  # Boost saturation
    hsv_image[..., 2] = cv2.add(hsv_image[..., 2], redness_boost, mask=red_mask)  # Boost brightness

    # Convert back to BGR
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Sharpness adjustment using a kernel
    kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 25, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return image

def detect_face(image):
    global detected_face_x, detected_face_y, detected_face_width, detected_face_height
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform face detection
    faces = detector.detect_faces(image_rgb)
    
    # Draw bounding boxes
    for face in faces:
     x, y, width, height = face['box']
     confidence = face['confidence']
     detected_face_x=x
     detected_face_y=y
     detected_face_width=width
     detected_face_height=height
     cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    return faces


def detect_eyes_lips(image):
   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
   
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
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
            cv2.rectangle(image, (x_min-5, y_min-5), (x_max+5, y_max+5), (255, 255, 255), -1)

        # Remove eyes with eyebrows
        draw_white_rectangle(eye_landmarks + eyebrow_landmarks)
        
        # Remove lips
        draw_white_rectangle(lip_landmarks)
    
   return image


# Main function to load image, apply filters, detect faces, and mark acne in a specific region
def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at '{image_path}'")
        return
    
    result = {}
    severety_level = ''
    suggested_medicine=[]

     
    # Load the Haar cascade model for face detection
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    # faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    # faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=7, minSize=(50, 50))
    # print(len(faces))


    faces = detect_face(image)

    if(len(faces)>1):
        result = {"message": "Multiple faces Detected","isFace":len(faces)}
        return result
    
    if(len(faces)==0):
        result = {"message": "No faces Detected","isFace":len(faces)}
        return result
    


    # Process each detected face
    # for (x, y, w, h) in faces:
    #     # Define a specific rectangular region within the face
    #     # roi_x_start, roi_x_end = x + 70, x + w - 70
    #     roi_x_start, roi_x_end = x , x + w 
    #     roi_y_start, roi_y_end = y, y + h
        # face_roi = filtered_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Detect acne marks within the defined rectangle
    response = detect_acne_marks(image)
        

    medicine_suggestions = {
    5: ['Isotretinoin', 'Clindamycin Gel','Consult a dermetologist'],
    4: ['Doxycycline', 'Adapalene Gel'],
    3: ['Benzoyl Peroxide', 'Salicylic Acid'],
    2: ['Niacinamide', 'Azelaic Acid'],
    1: ['Tea Tree Oil', 'Mild Cleanser'],
    0: ['No need to take any medication']
}

    # Determine severity level and suggest medicines
    if response > 75:
        severety_level = 5
    elif response > 60:
        severety_level = 4
    elif response > 45:
        severety_level = 3
    elif response > 30:
        severety_level = 2
    elif response > 15:
        severety_level = 1
    else:
        severety_level = 0
     
    
    
    suggested_medicine = medicine_suggestions[severety_level]
    
    result = {"message": "Acne Detected","isFace":len(faces),"Detected_acne_marks":response,"severety_level":severety_level,"suggested_medicine":suggested_medicine}
    print(result)
    return result

        # Place the processed region back into the main image
        # filtered_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = processed_face_roi

        # Draw the rectangle outline on the main image
        # cv2.rectangle(filtered_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

    # Display the final image with detected faces and acne marks
    # cv2.imshow('Detected Faces with Acne Marks', filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Print the number of faces found
    # print(f"Found {len(faces)} face(s) in the image.")

# Run the function with the path to your image
# main('acne9.jpg')












# //////////////////////////////////////////////////////



# Function to detect acne marks in a specific region of a face
# def detect_acne_marks(face_roi):
#     # Convert face ROI to HSV color space for color-based thresholding
#     hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

#     # Define multiple HSV ranges for red and pinkish tones
#     hsv_ranges = [
#         (np.array([0, 50, 50]), np.array([20, 255, 255])),    # Red tones
#         (np.array([140, 50, 50]), np.array([180, 255, 255]))  # Pink tones
#     ]

#     # Initialize an empty mask
#     final_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

#     # Apply each HSV range and combine masks
#     for lower_range, upper_range in hsv_ranges:
#         mask = cv2.inRange(hsv, lower_range, upper_range)
#         final_mask = cv2.bitwise_or(final_mask, mask)

#     # Find contours of potential acne marks in the combined mask
#     contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     counter=0

#     # Draw rectangles around detected acne areas on the face ROI
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if((x+w) > (x+3) and(x+w) < (x+20) and (y+h) < (y+20) ):
#          counter+=1
#         cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)   # Red rectangles

#     cv2.imshow('Detected Faces with Acne Marks', face_roi)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return counter

# # Function to apply filters to enhance the image
# def apply_filters(image, brightness=-100, contrast=250, saturation=80, sharpness=100):
#     from PIL import Image, ImageEnhance
#     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Brightness adjustment
#     enhancer = ImageEnhance.Brightness(pil_image)
#     pil_image = enhancer.enhance(1 + brightness / 255.0)

#     # Contrast adjustment
#     enhancer = ImageEnhance.Contrast(pil_image)
#     pil_image = enhancer.enhance(1 + contrast / 255.0)

#     # Convert back to OpenCV format
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

#     # Saturation adjustment
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation)
#     image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

#     # Sharpness adjustment using a kernel
#     kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 25, -1], [-1, -1, -1]])
#     image = cv2.filter2D(image, -1, kernel)

#     return image
