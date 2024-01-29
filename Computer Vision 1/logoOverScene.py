import cv2
import numpy as np


def superpose_images(scene_path, logo_path, result_path, scale_factor=0.5):
    scene = cv2.imread(scene_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    logo = cv2.resize(logo, (scene.shape[1], scene.shape[0]))

    alpha_channel = logo[:, :, 3] / 255.0

    inverse_alpha = 1 - alpha_channel

    for c in range(0, 3):
        scene[:, :, c] = (scene[:, :, c] * inverse_alpha +
                          logo[:, :, c] * alpha_channel).astype(np.uint8)

    result_scaled = cv2.resize(scene, (int(scale_factor * scene.shape[1]), int(scale_factor * scene.shape[0])))

    cv2.imwrite(result_path, result_scaled)
    cv2.imshow('Superposed Image', result_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


scene_image_path = 'images/input/scene.png'
logo_image_path = 'images/input/logo.png'
result_image_path = 'images/output/scene_logo.png'

# Example usage with a scale factor of 0.5 (you can adjust this value)
superpose_images(scene_image_path, logo_image_path, result_image_path)
