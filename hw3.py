import numpy as np

import colorsys
from PIL import Image
import os
import subprocess
import cv2
from sklearn.cluster import KMeans
import random

IM_PATHS = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg",
            "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg"]
BMP_IMS_DIR = "input_bmp_ims"
DEFAULT_SPATIAL_PROXIMITY_WEIGHT = 10  # [1, 30]
DEFAULT_NO_OF_SUPERPIXELS = 200  # [2, number of pixels]
SLIC_OUT_PATH = "slic_outputs/"
COLORED_IM_OUT_PATH = "colored_spx_ims/"
IM_WIDTH = 756
IM_HEIGHT = 502
NO_OF_IMAGES = 10
NO_OF_GABOR_FILTERS = 16
K = 10


def load_all_ims(im_paths):
    print("Loading images with paths: ")
    print(im_paths)
    ims = []
    for im_path in im_paths:
        im = load_im(im_path)
        ims.append(im)

    return ims


def load_im(im_path) -> np.ndarray:
    im = cv2.imread(im_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im_rgb


def _convert_ims_to_bmp(im_paths, out_dir):
    print("Converting images to .bmp format for SLIC algorithm...")
    # Images already exist
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) == 10:
        print("Aborted converting because the images already exist in .bmp format.")
        return

    # Create outdir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Save images as .bmp
    for path in im_paths:
        im = Image.open(path)
        im.save(f"{out_dir}/{path[:-4]}.bmp", "BMP")
    print("Finished converting.\n")


def run_slic(slic_exec_path, filename, spatial_proximity_weight, number_of_superpixels, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Create the command to run the executable
    command = [
        slic_exec_path,
        filename,
        str(spatial_proximity_weight),
        str(number_of_superpixels),
        results_path
    ]

    # Run the executable
    subprocess.run(command)


def read_labels(im_shapes) -> np.ndarray:
    print("\n\nReading labels...")

    labels_lst = []
    for i in range(1, 11):
        filename = f"slic_outputs/{i}.dat"
        with open(filename, "rb") as f:
            data = f.read()

        height = im_shapes[i-1][0]
        width = im_shapes[i-1][1]

        sz = width * height
        vals = np.frombuffer(data, dtype=np.int32, count=sz)
        labels = np.reshape(vals, (height, width))
        labels_lst.append(labels)

    print(f"Number of labels: {len(labels_lst)}\n")
    return labels_lst


def read_gabor_filters(im_number: int):
    gabor_filters = []
    for i in range(1, NO_OF_GABOR_FILTERS + 1):
        gabor_filter = cv2.imread(
            f"gabor_output/{im_number}_filtered_{i}.jpg", cv2.IMREAD_GRAYSCALE)
        gabor_filters.append(gabor_filter)

    return gabor_filters


def read_all_gabor_filters():
    print("Reading all Gabor filter outputs...")

    gabor_filters_lst = []
    for i in range(1, NO_OF_IMAGES + 1):
        gabor_filters = read_gabor_filters(i)
        gabor_filters_lst.append(gabor_filters)

    print(f"Number of gabor filters: {len(gabor_filters_lst)}\n")
    return gabor_filters_lst


def compute_gabor_features(labels, gabor_filters):
    height, width = labels.shape
    gabor_feat_mat = np.zeros((len(np.unique(labels)), len(gabor_filters)))
    unique_els, counts = np.unique(labels, return_counts=True)
    for filter_index, gabor_filter in enumerate(gabor_filters):
        for y in range(height):
            for x in range(width):
                spx_id = labels[y, x]
                gabor_filter_mag = gabor_filter[y, x]
                gabor_feat_mat[spx_id, filter_index] += gabor_filter_mag

    counts = counts.reshape((counts.shape[0], 1))
    gabor_feat_mat = gabor_feat_mat / counts
    return gabor_feat_mat


def generate_pseudo_colors_for_clusters(k: int):
    '''
        - k: number of clusters
        Returns:
            colors: An array of length 'k', with each index corresponding to a particular unique color.
    '''
    print(
        f"Generating {k} many unique random colors for each superpixel cluster.")
    colors = []
    while len(colors) < k:
        # Generate a random RGB color
        rgb = [random.randint(0, 255) for _ in range(3)]

        # Convert RGB to HSV
        hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        # Check if the HSV color is unique
        if all(hsv not in colorsys.rgb_to_hsv(*c) for c in colors):
            colors.append(rgb)

    # Convert RGB colors to hex format
    bgr_colors = [(rgb[2], rgb[1], rgb[0]) for rgb in colors]
    return bgr_colors


def color_images(colors, ims, spx_labels, spx_cluster_labels):
    print("Coloring superpixels based on their cluster labels...")
    cluster_size = len(colors)
    if not os.path.exists(COLORED_IM_OUT_PATH):
        os.mkdir(COLORED_IM_OUT_PATH)

    for i, (im, spx_label) in enumerate(zip(ims, spx_labels)):
        height, width, _ = im.shape
        for y in range(height):
            for x in range(width):
                spx_id = spx_label[y, x]
                cluster_label = spx_cluster_labels[i, spx_id]
                color = colors[cluster_label]
                im[y, x] = color

        cv2.imwrite(COLORED_IM_OUT_PATH + str(i + 1) + ".jpg", im)


def main():
    # Read images and convert to .bmp format
    # Then, save them in ./input_ims directory
    _convert_ims_to_bmp(IM_PATHS, BMP_IMS_DIR)
    jpg_ims = load_all_ims(IM_PATHS)

    # Run a command line argument to call the SLIC executable with given input images
    print("Running superpixel calculation...")
    bmp_im_paths = os.listdir(BMP_IMS_DIR)
    for im_name in bmp_im_paths:
        im_path = f"{BMP_IMS_DIR}/{im_name}"
        run_slic("SLICSuperpixelSegmentation.exe", im_path, DEFAULT_SPATIAL_PROXIMITY_WEIGHT,
                 DEFAULT_NO_OF_SUPERPIXELS, SLIC_OUT_PATH)

    # Assuming gabor filter is run on the original images,  compute Gabor features
    # for all superpixels by simply computing the average of Gabor features of the
    # pixels inside a particular superpixel.
    im_shapes = [(im.shape[0], im.shape[1]) for im in jpg_ims]
    labels_lst = read_labels(im_shapes)
    gabor_filters_lst = read_all_gabor_filters()

    # Compute Gabor features for all superpixels for all images
    print("Computing Gabor features for all superpixels for each image")
    gabor_feat_matrix_lst = []
    max_v_len = 0
    for gabor_filters, im_labels in zip(gabor_filters_lst, labels_lst):
        gabor_feat_matrix = compute_gabor_features(im_labels, gabor_filters)
        v_len = gabor_feat_matrix.shape[0]
        if v_len > max_v_len:
            max_v_len = v_len
        gabor_feat_matrix_lst.append(gabor_feat_matrix)

    # Apply k-means clustering
    row_adjusted_gabor_feat_mat_lst = []
    for i, gabor_feat_matrix in enumerate(gabor_feat_matrix_lst):
        v_len = gabor_feat_matrix.shape[0]
        width = gabor_feat_matrix.shape[1]
        if v_len < max_v_len:
            no_of_extra_rows_needed = max_v_len - v_len
            extra_rows = np.zeros((no_of_extra_rows_needed, width))
            new_arr = np.vstack((gabor_feat_matrix, extra_rows))
            row_adjusted_gabor_feat_mat_lst.append(new_arr)
        else:
            row_adjusted_gabor_feat_mat_lst.append(gabor_feat_matrix)

    k = K
    row_adjusted_gabor_feat_mat_lst = np.array(row_adjusted_gabor_feat_mat_lst)
    reshaped_row_adjusted_gabor_feat_mat_lst = row_adjusted_gabor_feat_mat_lst.reshape(
        -1, row_adjusted_gabor_feat_mat_lst.shape[-1])
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_row_adjusted_gabor_feat_mat_lst)

    cluster_labels = kmeans.labels_
    spx_cluster_labels = cluster_labels.reshape(
        (row_adjusted_gabor_feat_mat_lst.shape[0], row_adjusted_gabor_feat_mat_lst.shape[1]))
    centers = kmeans.cluster_centers_
    print("\n=====Superpixel Labels=====")
    print(spx_cluster_labels)
    print(f"Superpixel Labels shape: {spx_cluster_labels.shape}")
    print("\n=====Cluster centers=====")
    print(centers)
    print(f"Centers shape: {centers.shape}")

    colors = generate_pseudo_colors_for_clusters(k)
    print("Colors:")
    print(colors)

    # Color superpixels in images
    color_images(colors, jpg_ims, labels_lst, spx_cluster_labels)


main()
