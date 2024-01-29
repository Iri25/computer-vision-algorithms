import cv2
import os
import numpy as np


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images[filename] = image
    return images

def compare_histograms(base_img, images, method):
    base_hist = cv2.calcHist([base_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # base_hist = cv2.normalize(base_hist, base_hist).flatten()
    cv2.normalize(base_hist, base_hist, 0, 1, cv2.NORM_MINMAX)

    results = {}
    for (k, image) in images.items():
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # hist = cv2.normalize(hist, hist).flatten()
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        d = cv2.compareHist(base_hist, hist, method)
        results[k] = d

    return results


def color_reduce(image, k=8):
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    reduced_image = centers[labels.flatten()].reshape(image.shape)
    return reduced_image


# Load sources
image_dataset = load_images_from_folder('images')
input_image = cv2.imread('circle_green.png')

# Standard histogram comparison
for metric in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]:
    standard_results = compare_histograms(input_image, image_dataset, metric)
    print("Histogram Comparison " + str(metric))
    print('---------------------------------------------------')
    for i in standard_results:
        print(i + ': similarity ' + str(standard_results[i]))
    print()

print()
print('####################################################')
print()

# Color-reduced histogram comparison
for metric in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]:
    reduced_input_image = color_reduce(input_image)
    reduced_dataset = {k: color_reduce(v) for k, v in image_dataset.items()}
    reduced_results = compare_histograms(reduced_input_image, reduced_dataset, cv2.HISTCMP_CHISQR)
    print("Reduced Color Histogram Comparison " + str(metric))
    print('---------------------------------------------------')
    for i in reduced_results:
        print(i + ': color reduced ' + str(reduced_results[i]))
    print()

