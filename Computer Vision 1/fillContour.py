import cv2
import numpy as np


def is_similar(image1, image2):
    # bitwise_xor function calculates the per-element bit-wise exclusive-OR of two arrays or an array and a scalar
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())


image = cv2.imread('images/input/scene.png')

if image is None:
    print("Image cannot be found")
    exit()

cv2.imshow("Initial image", image)
cv2.imwrite('images/output/initial_image.png', image)

# convert image to greyscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", image)
cv2.imwrite('images/output/gray_image.png', image)

# get image dimensions
rows, cols = image.shape

# select starting point - start node
start_x = int(rows / 2)
start_y = int(cols / 2)

empty_canvas = np.zeros((rows, cols), np.uint8)
# matrix to be used in dilation
# np.uint8 - construct an 8-bit unsigned integer
# empty_canvas will return an array of zeros with the given shape of the initial image
# kernel will return a new array of given shape and type(3,3), filled with ones.

# get contour
ret, thresh = cv2.threshold(image, 40, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours is a Python list of all the contours in the image.
# each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
# hierarchy is a full family hierarchy list for the image

# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

cv2.imshow('Contour', image_copy)
cv2.imwrite('images/output/contour_image.png', image_copy)

# set the new color
empty_canvas[start_x, start_y] = 255  # choose the replacement color

img_complementary = cv2.bitwise_not(image)
cv2.imshow("Image complementary", img_complementary)
cv2.imwrite('images/output/complementary_image.png', img_complementary)

# get point
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))

while True:
    while True:
        img_temp = cv2.dilate(empty_canvas, cross_kernel)

        # intersect temp_img with complementary
        temp_canvas = cv2.bitwise_and(img_temp, img_complementary, mask=img_complementary)

        # check if contour is done - if the empty canvas from beginning is the same as temp one-we've been everywhere
        if is_similar(temp_canvas, empty_canvas):
            break
        empty_canvas = temp_canvas

    img_processed = cv2.add(image, empty_canvas)
    cv2.imshow('Final result', img_processed)
    cv2.imwrite('images/output/final_image.png', img_processed)

    if cv2.waitKey(1) == ord("x"):
        break

cv2.destroyAllWindows()
