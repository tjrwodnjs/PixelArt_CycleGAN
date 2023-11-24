import numpy as np
import matplotlib.pyplot as plt
import sys
from dezero import cuda

def DiffAugment(x, policy='', channels_first=True, is_cuda=False):
    if policy:
        if not channels_first:
            x = x.transpose(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, is_cuda=is_cuda)
        if not channels_first:
            x = x.transpose(0, 2, 3, 1)
        if is_cuda:
            x = cuda.as_cupy(x).copy()
            return x
        x = x.copy()
    return x

def rand_brightness(x, is_cuda=False):
    if is_cuda:
        x = x + (cuda.cupy.random.rand(x.shape[0], 1, 1, 1, dtype=x.dtype) - 0.5)
        return x

    x = x + (np.random.rand(x.shape[0], 1, 1, 1) - 0.5)
    return x

def rand_saturation(x, is_cuda=False):
    if is_cuda:
        x_mean = cuda.as_cupy(x).mean(axis=3, keepdims=True)
        x = (x - x_mean) * (cuda.cupy.random.rand(x.shape[0], 1, 1, 1, dtype=x.dtype) * 2) + x_mean
        return x

    x_mean = x.average(axis=3, keepdims=True)
    x = (x - x_mean) * (np.random.rand(x.shape[0], 1, 1, 1) * 2) + x_mean
    return x

def rand_contrast(x, is_cuda=False):
    if is_cuda:
        magnitude = cuda.cupy.random.rand(x.shape[0], 1, 1, 1, dtype=x.dtype) + 0.5
        x_mean = cuda.as_cupy(x).mean(axis=(1, 2, 3), keepdims=True)
        x = (x - x_mean) * magnitude + x_mean
        return x
    
    magnitude = np.random.rand(x.shape[0], 1, 1, 1) + 0.5
    x_mean = x.mean(axis=(1, 2, 3), keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x

# def make_rgb_to_gray(x):
#     x = x[:, 0:1, :, :] * 0.2989 + x[:, 1:2, :, :] * 0.5870 + x[:, 2:3, :, :] * 0.1140
#     return x

def make_rgb_to_gray(x):
    x = np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])
    # x = np.mean(x, axis=3, keepdims=True)
    return x

def make_rgb_to_black(batch, threshold=0.85):
    # Define a white threshold (adjust as needed)
    white_threshold = threshold # You can adjust this threshold value

    # Convert the batch of images to grayscale (if they are not already)
    batch_gray = make_rgb_to_gray(batch)

    # Create a mask where non-white pixels are set to black
    mask = batch_gray < white_threshold

    # Set non-white pixels to black
    batch[mask] = [0, 0, 0]  # Assuming the images are in RGB format

    return batch


def rand_cutout(images, ratio=0.5, is_cuda=False):
    """
    Apply random cutout to a batch of NumPy images.

    Args:
        images (numpy.ndarray): Batch of input images as a NumPy array with shape (batch_size, H, W, C).
        ratio (float): Ratio of the image size to be cut out.

    Returns:
        numpy.ndarray: Batch of images with random cutout applied.
    """
    batch_size, h, w, c = images.shape

    # Calculate the cutout size based on the ratio
    cutout_h = int(h * ratio)
    cutout_w = int(w * ratio)

    if is_cuda:
        top = cuda.cupy.random.randint(0, h - cutout_h + 1, size=batch_size)
        left = cuda.cupy.random.randint(0, w - cutout_w + 1, size=batch_size)
        mask = cuda.cupy.ones((batch_size, h, w, c), dtype=images.dtype)
        for i in range(batch_size):
            mask[i, top[i]:top[i]+cutout_h, left[i]:left[i]+cutout_w, :] = 0
        result_images = images * mask
        return result_images
    # Generate random coordinates for the top-left corner of the cutout for each image in the batch
    top = np.random.randint(0, h - cutout_h + 1, size=batch_size)
    left = np.random.randint(0, w - cutout_w + 1, size=batch_size)

    # Create a mask for each image in the batch to zero out the cutout region
    mask = np.ones((batch_size, h, w, c), dtype=images.dtype)
    for i in range(batch_size):
        mask[i, top[i]:top[i]+cutout_h, left[i]:left[i]+cutout_w, :] = 0

    # Apply the mask to the batch of images
    result_images = images * mask

    return result_images

AUGMENT_FNS = {
    'color': [rand_brightness, rand_contrast],
    # 'translation': [rand_translation],
    'cutout': [rand_cutout],
}