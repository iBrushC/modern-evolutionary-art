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


# L2 error with added random noise
def seasoned_L2(a: np.ndarray, b: np.ndarray, intensity: float=1, power: float=2) -> float:
    if (a.shape != b.shape): return -1
    salt = (np.random.rand(*a.shape) - 0.5) ** power
    salt *= intensity
    
    return np.linalg.norm(a - b - salt)
