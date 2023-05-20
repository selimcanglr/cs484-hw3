import numpy as np
from skimage.color import rgb2lab
import cv2


def _read_im(path) -> np.ndarray:
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def get_pixel_details(image, x, y):
    rgb = image[y, x]
    l, a, b = rgb2lab([[rgb]])[0, 0]
    pixel = {
        "l": l,
        "a": a,
        "b": b,
        "x": x,
        "y": y,
    }
    return pixel


def initialize_cluster_centers(image, step_size):
    cluster_centers = []
    height, width, _ = image.shape

    for y in range(step_size // 2, height, step_size):
        for x in range(step_size // 2, width, step_size):
            cluster_center = get_pixel_details(image, x, y)
            cluster_centers.append(cluster_center)

    return cluster_centers


def compute_distance(center, pixel, S, m):
    # m is in the range [0, 40]
    # When m is large, spatial proximity is more important and the
    # resulting superpixels are more compact (i.e., they have a lower
    # area to perimeter ratio). When m is small, the resulting superpixels
    # adhere more tightly to image boundaries, but have less regular
    # size and shape.
    l_diff_squared = np.square(center["l"] - pixel["l"])
    a_diff_squared = np.square(center["a"] - pixel["a"])
    b_diff_squared = np.square(center["b"] - pixel["b"])
    D_C = np.sqrt(l_diff_squared + a_diff_squared + b_diff_squared)

    x_diff_squared = np.square(center["x"] - pixel["x"])
    y_diff_squared = np.square(center["y"] - pixel["y"])
    D_S = np.sqrt(x_diff_squared + y_diff_squared)

    D = np.sqrt(np.square(D_C) + np.square(D_S / S) * np.square(m))
    return D


def SLIC(image: np.ndarray, K: int, m: int = 20, iterations: int = 10):
    # INITIALIZATION
    # Initialize cluster centers
    height, width, _ = image.shape
    N = image.size  # Total pixels
    S = int(np.sqrt(N / K))  # Step size for initializing cluster centers
    cluster_centers = initialize_cluster_centers(image, S)

    # TODO: Move cluster centers to the lowest gradient position in a 3x3 neighborhood

    # Set labels to -1
    labels = np.ones(image.shape) * -1

    # Set distances to infinity
    distances = np.ones(image.shape) * np.inf

    i = 0
    while i < iterations:
        # ASSIGNMENT
        for k, center in enumerate(cluster_centers):
            center_x, center_y = center["x"], center["y"]
            # For each pixel in a 2Sx2S region around the center
            offset = 2 * S
            for y in range(center_y - offset, center_y + offset):
                for x in range(center_x - offset, center_x + offset):
                    if y >= 0 and y < image.shape[0] and x >= 0 and x < image.shape[1]:
                        pixel = get_pixel_details(image, x, y)
                        D = compute_distance(center, pixel, S, m)
                        if D < distances[y, x]:
                            distances[y, x] = D
                            labels[y, x] = k

        # UPDATE
        # Compute new cluster centers
        for k, center in enumerate(cluster_centers):
            pixels = []
            for y in range(height):
                for x in range(width):
                    if labels[y, x] == k:
                        pixel = get_pixel_details(image, x, y)
                        pixel = [pixel["l"], pixel["a"],
                                 pixel["b"], pixel["x"], pixel["y"]]
                        pixels.append(pixel)

            mean_vector = np.mean(pixels, axis=0)
            cluster_centers[k] = mean_vector

        i += 1

    return labels


def main():
    im = _read_im("./1.jpg")
    SLIC(im, 20)


main()
