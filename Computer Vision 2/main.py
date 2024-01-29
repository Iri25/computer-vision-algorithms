import cv2
import numpy as np


def corners_detection(image):

    # cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    cross = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]], dtype=np.uint8)

    # diamond = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    diamond = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)
    # diamond = np.array([[0, 1, 0],
    #                     [1, 1, 1],
    #                     [0, 1, 0]], dtype=np.uint8)

    x_shape = np.array([[1, 0, 0, 0, 1],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 1]], dtype=np.uint8)

    square = np.ones((5, 5), np.uint8)

    # first dilation
    R1 = cv2.dilate(image, cross)
    R1 = cv2.erode(R1, diamond)
    cv2.imshow("First result", R1)
    cv2.imwrite("images/output/first_dilate_erode_image.png", R1)

    # second dilation
    R2 = cv2.dilate(image, x_shape)
    R2 = cv2.erode(R2, square)
    cv2.imshow("Second result", R2)
    cv2.imwrite("images/output/second_dilate_erode_image.png", R2)

    result = cv2.absdiff(R1, R2)
    cv2.imshow("No threshold", result)
    cv2.imwrite("images/output/no_threshold_image.png", result)

    retval, result = cv2.threshold(result, 70, 255, cv2.THRESH_BINARY)

    cv2.imshow("Final result", result)
    cv2.imwrite("images/output/final_result_image.png", result)


image_path = cv2.imread('images/input/building.png')
if image_path is None:
    print("Image cannot be found")
    exit()

corners_detection(image_path)

while True:
    if cv2.waitKey(1) == ord("x"):
        break

cv2.destroyAllWindows()
