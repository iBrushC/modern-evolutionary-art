import numpy as np
import cv2

# Fits image into a square based on the largest dimension
def fit_image_to_square(image: np.ndarray, n: int, background: int=1) -> np.ndarray:
    (height, width) = image.shape[0], image.shape[1]
    if (width == height): return cv2.resize(image, (n, n))

    if (width > height):
        pad = (width - height) // 2
        square = cv2.copyMakeBorder(image, top=pad, bottom=pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=background)
        return cv2.resize(square, (n, n))
    else:
        pad = (height - width) // 2
        square = cv2.copyMakeBorder(image, left=pad, right=pad, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT, value=background)
        return cv2.resize(square, (n, n))

# Root Mean Squared Error
def RMSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(
        np.mean(
            (a.astype(np.float32) - b.astype(np.float32))**2
        )
    )

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
        canvas = cv2.line(canvas, line[0].astype(np.int32), line[1].astype(np.int32), 0, thickness)

    return canvas

# Draw lines from an array (x1, y1, x2, y2)
def normalized_lines_points(points: np.ndarray, dimensions: tuple, thickness: int=1) -> np.ndarray:
    canvas = np.ones(dimensions)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        start = (line[0] * dimensions).astype(np.int32)
        end = (line[1] * dimensions).astype(np.int32)
        canvas = cv2.line(canvas, start, end, 0, thickness)

    return canvas

# Draw lines from an array (x, y, rotation, length)
def normalized_lines_center(points: np.ndarray, dimensions: tuple, thickness: int=1) -> np.ndarray:
    canvas = np.ones(dimensions)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        center = line[0]
        offset = np.array([np.cos(line[1][0] * 2*np.pi), np.sin(line[1][0] * 2*np.pi)]) * line[1][1] * 0.25 # 0.25 is the scaling factor to prevent ugly screen-spanning lines
        start = (center + offset) * dimensions
        end = (center - offset) * dimensions
        canvas = cv2.line(canvas, start.astype(np.int32), end.astype(np.int32), 0, thickness)

    return canvas

# Draw lines with opacity from an array
def opacity_normalized_lines_points(points: np.ndarray, dimensions: tuple, thickness: int=1, opacity=0.5) -> np.ndarray:
    canvas = np.ones(dimensions)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        start = (line[0] * dimensions).astype(np.int32)
        end = (line[1] * dimensions).astype(np.int32)
        overlay = cv2.line(np.zeros(dimensions), start, end, opacity, thickness)
        canvas -= overlay

    return np.clip(canvas, 0, 1)

# Draw lines with opacity from an array
def opacity_normalized_lines_center(points: np.ndarray, dimensions: tuple, thickness: int=1, opacity=0.5) -> np.ndarray:
    canvas = np.ones(dimensions)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        center = line[0]
        offset = np.array([np.cos(line[1][0] * 2*np.pi), np.sin(line[1][0] * 2*np.pi)]) * line[1][1] * 0.25
        start = (center + offset) * dimensions
        end = (center - offset) * dimensions
        overlay = cv2.line(
            np.zeros(dimensions), 
            start.astype(np.int32), 
            end.astype(np.int32), 
            opacity,
            thickness
        )
        canvas -= overlay

    return np.clip(canvas, 0, 1)

# Draw lines with opacity from an array
def opacity_normalized_lines_center_bg(points: np.ndarray, background: np.ndarray, thickness: int=1, opacity=0.5) -> np.ndarray:
    canvas = np.array(background, dtype=np.float64)
    lines = points.reshape((-1, 2, 2))
    
    for line in lines:
        center = line[0]
        offset = np.array([np.cos(line[1][0] * 2*np.pi), np.sin(line[1][0] * 2*np.pi)]) * line[1][1] * 0.25
        start = (center + offset) * background.shape[0:2]
        end = (center - offset) * background.shape[0:2]
        overlay = cv2.line(
            np.zeros(background.shape[0:2]), 
            start.astype(np.int32), 
            end.astype(np.int32), 
            opacity,
            thickness
        )
        canvas -= overlay

    return np.clip(canvas, 0, 1)