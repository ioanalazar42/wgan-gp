'''Contains utilities for reading and processing images.'''

import numpy as np
import os

from skimage import io, transform


def _center_crop_image(image):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = height if height < width else width

    y = int((height - crop_size) / 2)
    x = int((width - crop_size) / 2)

    return image[y : crop_size, x : crop_size]


def _resize_image(image, width, height):
    return transform.resize(image, [height, width, 3], anti_aliasing=True, mode='constant')


def _mean_normalize(image):
    '''Takes an image with float values between [0, 1] and normalizes it to [-1, 1]'''
    return 2 * image - 1


def _load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        # Convert grayscale images to RGB
        # (yes you can actually save a JPEG without RGB info (so like 500x500 instead of 500x500x3), *mind blown*)
        print('Image "{}" is grayscale!'.format(path))
        image = np.dstack([image, image, image])

    image = _mean_normalize(_resize_image(_center_crop_image(image), 128, 128))

    # Change the 128x128x3 image to 3x128x128 as expected by PyTorch.
    return image.transpose(2, 0, 1)


def load_images(dir_path):
    file_names = os.listdir(dir_path)
    images = np.empty([len(file_names), 3, 128, 128], dtype=np.float32)
    print('Loading {} images from {}...'.format(len(file_names), dir_path))

    for i, file_name in enumerate(file_names):
        image_path = os.path.join(dir_path, file_name)
        images[i] = _load_image(image_path)

        if i > 0 and i % 10000 == 0:
            print('Loaded {}/{} images so far'.format(i, len(images)))

    return images
