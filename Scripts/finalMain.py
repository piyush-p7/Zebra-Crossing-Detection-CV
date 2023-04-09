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
    gray_norm = np.divide(gray, 255)

    # Apply Gaussian smoothing
    gray_blur = cv2.GaussianBlur(gray_norm, (5, 5), 70)

    # Apply thresholding
    thresh = cv2.threshold(gray_blur, 0.7, 255, cv2.THRESH_BINARY)[1]

    # Apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,10))
    morph = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Find connected components
    num_crossings, labels = cv2.connectedComponents(morph)

    # Subtract 1 from the number of crossings to exclude the background component
    num_crossings -= 1
    
    # Draw the contours
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.drawContours(img.copy(), contours, -2, (255, 0, 0), 2)

    # Save the image with contours
    img_file_name = img_file
    img_file_path = os.path.join(preprocessed_dir, img_file_name)
    cv2.imwrite(img_file_path, img_contours)

    # Print the number of crossings
    print(f"Number of zebra crossings: {num_crossings}")
