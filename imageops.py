import numpy as np
import cv2

# Fits image into a square based on the largest dimension
def fit_image_to_square(image: np.ndarray, n: int, background: int=255) -> np.ndarray:
    (height, width) = image.shape[0], image.shape[1]
    if (width == height): return image

    if (width > height):
        pad = (width - height) // 2
        square = cv2.copyMakeBorder(image, top=pad, bottom=pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=background)
        return cv2.resize(square, (n, n))
    else:
        pad = (height - width) // 2
        square = cv2.copyMakeBorder(image, left=pad, right=pad, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT, value=background)
        return cv2.resize(square, (n, n))

# L2 error
def L2(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a.astype(np.int32) - b.astype(np.int32))

# L2 error with added random noise
def seasoned_L2(a: np.ndarray, b: np.ndarray, intensity: float=1) -> float:
    if (a.shape != b.shape): raise ValueError("Invalid comparison shapes", "The shape of arrays a and b are not equal")
    salt = (np.random.rand(*a.shape) - 0.5) * intensity
    
    return np.linalg.norm(a - b - salt)

# Random cropped L2 which gives better error (???) but worse performance
def crop_seasoned_L2(a: np.ndarray, b: np.ndarray, intensity: float=1, iterations: int=3, percent_max_offset: float=0.3) -> float:
    if (a.shape != b.shape): raise ValueError("Invalid comparison shapes", "The shape of arrays a and b are not equal")

    percent_max_offset = np.clip(percent_max_offset, 0, 0.5)
    dimensions = np.array(a.shape)
    running_sum = 0

    for i in range(iterations):
        top_left = dimensions * percent_max_offset * np.random.rand(len(a.shape))
        top_left = top_left.astype(np.int32)
        bottom_right = dimensions - dimensions * percent_max_offset * np.random.rand(len(a.shape))
        bottom_right = bottom_right.astype(np.int32)

        running_sum += seasoned_L2(a[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], b[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], intensity)

    return running_sum / iterations

# Draw lines from an array
def draw_lines(points: np.ndarray, dimensions: tuple, thickness: int=1) -> np.ndarray:
    canvas = np.ones(dimensions)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        canvas = cv2.line(canvas, line[0], line[1], 0, thickness)

    return canvas

# Draw lines from an array
def draw_normalized_lines(points: np.ndarray, dimensions: tuple, thickness: int=1) -> np.ndarray:
    canvas = np.ones(dimensions)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        start = (line[0] * dimensions).astype(np.int32)
        end = (line[1] * dimensions).astype(np.int32)
        canvas = cv2.line(canvas, start, end, 0, thickness)

    return (canvas * 255).astype(np.uint8)
