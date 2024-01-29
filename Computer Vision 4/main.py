import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_lines(image_path, rho, theta, threshold):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use HoughLines method to detect lines
    lines = cv2.HoughLines(edges, rho, theta, threshold)

    # Draw lines on the original image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    plt.subplot(121), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Line Detection'), plt.xticks([]), plt.yticks([])

    plt.show()


def detect_circles(image_path, dp, min_dist, param1, param2, min_radius, max_radius):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply HoughCircles method to detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the result
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Circle Detection'), plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        print("No circles detected.")


# Example usage:
# Adjust the parameters as needed for your specific images
detect_lines('Icosahedron.jpg', rho=1, theta=np.pi / 180, threshold=50)
detect_circles('Icosahedron.jpg', dp=1, min_dist=50, param1=50, param2=30, min_radius=0, max_radius=0)
