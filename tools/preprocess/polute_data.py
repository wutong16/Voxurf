import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
import torch
import os
from PIL import Image

# define a function for
# compressing an image
def compressMe(filename, output_path, quality=50):

    # open the image
    img = Image.open(filename)

    img.save(output_path,
             "JPEG",
             optimize = True,
             quality = quality)
    return img

def add_motion_blur(filename, output_path, kernel_size=15):

    img = cv2.imread(filename)

    # Specify the kernel size.
    # The greater the size, the more the motion.

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)

    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)

    cv2.imwrite(output_path, horizonal_mb)

    return horizonal_mb

def lower_resolution(filename, output_path, down_ratio=4):
    img = cv2.imread(filename)
    h, w, _ = img.shape
    img = cv2.resize(img, (w//down_ratio, h//down_ratio), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, img)

    return img

def polute_folder(input_dir, output_dir, mode='motion_blur'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    for file in files:
        if not file.endswith(".png") and not file.endswith(".jpg"):
            continue
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        if mode == 'motion_blur':
            output = add_motion_blur(input_path, output_path)
        elif mode == 'down_res':
            output = lower_resolution(input_path, output_path)
        elif mode == 'compression':
            output = compressMe(input_path, output_path, quality=40)
    print("done for {}".format(input_dir))

if __name__ == "__main__":
    # polute_folder("../public_data/dtu_scan122/image", "../public_data/dtu_scan122_mb/image")
    # polute_folder("../public_data/dtu_scan122/image", "../public_data/dtu_scan122_down8/image", mode="down_res")
    polute_folder("../public_data/dtu_scan122/image", "../public_data/dtu_scan122_compress40/image", mode="compression")
