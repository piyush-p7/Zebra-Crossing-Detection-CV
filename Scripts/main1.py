import cv2

# Load image
img = cv2.imread("./Images/ZCrossing.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply threshold using Otsu's method
thresh, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        cv2.drawContours(img, [contour], 0, (33, 255, 0), 2)
        num_contours += 1

# Print number of contours
print(f"Number of zebra crossing lines detected: {num_contours}")

# Display image with contours
cv2.imshow('Zebra Crossing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()