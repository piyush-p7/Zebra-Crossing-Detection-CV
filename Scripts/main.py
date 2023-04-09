import cv2
import numpy as np
import os

img_dir = r'./Images/'

# Create the directory to store preprocessed images if it doesn't exist
preprocessed_dir = r'./Preprocessed_Images/'

if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Loop over all the images in the directory
for img_file in os.listdir(img_dir):
    # Load the image
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    print(img_file ," Processing.......")
    

    # # Resize the image
    # img = cv2.resize(img, (800, 600))

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values
    gray_norm = np.divide(gray, 255)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray_norm, (5, 5), 5)

    # Convert the image to uint8 type
    blur_uint8 = (blur * 255).astype(np.uint8)

    # Apply threshold using Otsu's method
    thresh, binary = cv2.threshold(blur_uint8, 0.7, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological opening to remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Apply Canny edge detection algorithm
    edges = cv2.Canny(opening, 100, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image and count number of contours
    num_contours = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # adjust threshold to detect zebra crossing lines accurately
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            num_contours += 1

    # Print number of contours
    print("Number of zebra crossing lines detected: " + str(num_contours))
    print("Image Processed.\n")

    # Save the image with contours
    img_file_name = img_file
    img_file_path = os.path.join(preprocessed_dir, img_file_name)
    cv2.imwrite(img_file_path, img)

