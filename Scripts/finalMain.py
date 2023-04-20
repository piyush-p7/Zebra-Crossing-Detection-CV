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
    print(img_path)

    # Resize the image
    img = cv2.resize(img, (800, 600))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values
    gray_norm = np.divide(gray, 250)

    # Apply Gaussian smoothing
    gray_blur = cv2.GaussianBlur(gray_norm, (17, 17), 400)

    # Apply thresholding
    thresh = cv2.threshold(gray_blur, 0.8, 400, cv2.THRESH_BINARY)[1]

    # Apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    morph = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image and count number of contours
    num_contours = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 650:  # adjust threshold to detect zebra crossing lines accurately
            cv2.drawContours(img, [contour], -2, (0, 5,255), 2)
            num_contours += 1

    # Save the image with contours
    img_file_name = img_file
    img_file_path = os.path.join(preprocessed_dir, img_file_name)
    cv2.imwrite(img_file_path, img)

    # Print the number of crossings
    print(f"Number of zebra crossings: {num_contours}")
