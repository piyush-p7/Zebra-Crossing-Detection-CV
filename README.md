# Zebra Crossing Detection
This project is focused on detecting Zebra Crossings from input images. It uses Computer Vision techniques to preprocess the input image and detect Zebra Crossings in it.

# Dependencies
- Python 3
- OpenCV 4.5.3 or higher

# Getting Started
 - Clone this repository.
 - Navigate to the project directory.
 - Install the dependencies.
 - Run the script zebra_crossing_detection.py in a Python environment.

# Algorithm
- Read input image using OpenCV.
- Convert the image to grayscale.
- Enhance contrast with adaptive histogram equalization.
- Threshold the image using Otsu's binarization method.
- Apply morphological operations to clean up the image.
- Apply Gaussian blur to reduce noise.
- Detect edges using Canny edge detection.
- Find lines using HoughLinesP.
- Initialize variables for Zebra Crossing detection.
- Draw lines and count Zebra Crossings.
- Print the number of Zebra Crossings and display the output.

# Results
The output of the script includes the original image, the processed image, and the number of Zebra Crossings detected in the image.

# Future Work
The algorithm could be improved to detect Zebra Crossings in videos as well.
The output could be made more user-friendly by displaying a message indicating the presence or absence of Zebra Crossings in the image.
