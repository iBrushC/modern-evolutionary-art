import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import time

from imageops import *
from evolution import *

# Line thickness should be determined by a constant over the magnitude of the dimensions

def main():
    line_count = 120
    solution_size = 4 * line_count # Written to indicate (how many components per line) * (how many lines)
    square_size = 100

    target_image = cv2.imread("images/eye.jpg")
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    target_image = cv2.GaussianBlur(target_image, (1, 1), cv2.BORDER_DEFAULT)
    # dX = cv2.Sobel(target_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    # dY = cv2.Sobel(target_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    # target_image *= cv2.erode(1.0 - np.sqrt(dX**2 + dY**2), np.ones((3, 3)), iterations=1)
    target_image = (target_image * 255).astype(np.uint8)
    target_image = cv2.convertScaleAbs(target_image, alpha=1.5, beta=0)
    target_image = fit_image_to_square(image=target_image, n=square_size)

    # Error function
    def error_function(base: np.ndarray, lines: np.ndarray) -> float:
        lines = draw_normalized_lines(lines, base.shape, 1)
        lines = cv2.blur(lines, ksize=(2, 2))  # Blur simulates how lines close together simulate a lighter shade
        # error = seasoned_L2(base, lines, intensity=2.5)
        error = L2(base, lines)
        return error

    optimizer = AdamOptimizer(
        solution_size, 
        alpha=0.3, 
        beta1=0.8,
        beta2=0.95,
        decay=1,
        decay_alpha=True, 
        mode="minimize"
    ) # Alpha is one because the solver applies its own step size

    stdev = np.ones(solution_size) * 0.1
    solver = ESSolver(
        parameter_count=solution_size,
        population_count=125, 

        center=None,
        center_bounds=(0, 1),
        center_alpha=0.1,

        sigma=stdev,
        sigma_bounds=(0.01, 0.5),
        sigma_alpha=0.15,

        seed=9,
        optimizer=optimizer,
        error_function=lambda lines: error_function(target_image, lines)
    )
    solver.randomize_center(min=0.1, max=0.9)

    errors = solver.climb(cycles=2000, log_every=50)

    # Final logging
    print(f"\n\nDelta Error: {max(errors) - min(errors)}")
    print(f"Final L2: {error_function(target_image, solver.center)}")
    
    # Graphs
    plt.figure()
    plt.title("Error history")
    plt.plot(errors)

    plt.figure()
    plt.title("Target Image")
    plt.imshow(target_image, cmap="Greys_r")

    plt.figure()
    plt.title("Line Recreation")
    lines = draw_normalized_lines(solver.center, (square_size, square_size), 1)
    lines = cv2.blur(lines, ksize=(2, 2))
    plt.imshow(lines, cmap="Greys_r")

    plt.figure()
    plt.title("Line Recreation Upscaled")
    plt.imshow(draw_normalized_lines(solver.center, (square_size * 4, square_size * 4), 2), cmap="Greys_r")

    plt.show()
    

if __name__ == "__main__":
    main()