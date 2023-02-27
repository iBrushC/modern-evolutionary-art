import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import time

from imageops import *
from evolution import *
from gcode import *

LINE_THICKNESS = 1
LINE_OPACITY = 0.6

def main():
    lines_per_batch = 16
    batch_size = 4 * lines_per_batch # Four components per batch
    batches = 43
    seed = 794

    square_size = 200

    target_image = cv2.imread("images/skull.jpg")
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    target_image = fit_image_to_square(image=target_image, n=square_size, background=1)

    # Error function
    def batch_error(base: np.ndarray, lines: np.ndarray, background: np.ndarray) -> float:
        lines = opacity_normalized_lines_center_bg(lines, background, thickness=LINE_THICKNESS, opacity=LINE_OPACITY)
        lines = cv2.blur(lines, ksize=(7, 7))
        error = RMSE(base, lines)
        return error

    # Create the optimizer
    optimizer = AdamOptimizer(
        batch_size, 
        alpha=0.4, 
        beta1=0.9,
        beta2=0.99,
        decay=1,
        decay_alpha=True, 
        mode="minimize"
    )

    # Create a blank background canvas
    background = np.ones(shape=(target_image.shape), dtype=np.float32)
    
    # Create the evolution stratgies solver
    stdev = np.ones(batch_size) * 0.1
    essolver = ESSolver(
        parameter_count=batch_size,
        population_count=30, 

        center=None,
        center_bounds=(0, 1),
        center_alpha=0.1,

        sigma=stdev,
        sigma_bounds=(0.01, 0.5),
        sigma_alpha=0.1,

        seed=seed,
        optimizer=optimizer,
        error_function=lambda lines: batch_error(target_image, lines, background)
    )
    essolver.randomize_center(low=0, high=1)

    # Create the genetic solver
    gensolver = GeneticSolver(
        parameter_count=batch_size, 
        population_count=30, 
        mode="minimize", 
        mutation_probability=0.2, 
        mutation_strength=0.3,
        use_crossbreeding=True,
        seed=seed, 
        error_function=lambda lines: batch_error(target_image, lines, background)
    )
    gensolver.randomize_population(low=0, high=1)

    # This process uses a mini batch gradient descent that uses both genetic algorithms and evolution strategies in a batch.
    # The genetic algroithm is there to find a good starting solution, while the evolution strategies are there to optimize it.
    # This works very well if given enough cycles, but ends up being harmful if too many batches are given
    background = np.ones(shape=(target_image.shape))
    all_errors = []
    all_lines = np.zeros(shape=(batch_size * batches))
    for i in range(batches):
        # Re-randomize lines
        gensolver.randomize_population(low=0, high=1)

        # Climb log the errors
        print(f"\n\nBATCH {i+1}:")
        print("Genetic:")
        generrors = gensolver.climb(cycles=30, log_every=30)
        all_errors.extend(generrors)
        print("Evolutionary:")
        essolver.center = gensolver.best
        essolver.sigma = np.ones(batch_size) * 0.1
        eserrors = essolver.climb(cycles=100, offset=30, log_every=130)
        all_errors.extend(eserrors)

        # Store computed lines
        all_lines[(i * batch_size):((i+1) * batch_size)] = essolver.center

        # Set the background to be what was previously generated
        background = opacity_normalized_lines_center_bg(essolver.center, background, thickness=LINE_THICKNESS, opacity=LINE_OPACITY)

    # Save to GCODE
    generate_gcode_from_lines(all_lines, mode="center", name="Test", offset=(25, 25, 0), scale=(125, 125))

    # Show the target image
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title("Target")
    plt.imshow(target_image, cmap="Greys_r")
    plt.clim(0, 1)

    # Show what was created
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title("Generated")
    plt.imshow(background, cmap="Greys_r")
    plt.clim(0, 1)

    # Show a larger version
    upscale_multiplier = 2
    upscale_generated = opacity_normalized_lines_center_bg(
        all_lines, 
        np.ones(shape=(background.shape[0] * upscale_multiplier, background.shape[1] * upscale_multiplier)), 
        thickness=int(LINE_THICKNESS * upscale_multiplier / 2), 
        opacity=1
    )
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title("Generated Upscaled")
    plt.imshow(upscale_generated, cmap="Greys_r")
    plt.clim(0, 1)

    # Graph of errors
    plt.figure(figsize=(40,24))
    plt.title("Errors")
    plt.plot(all_errors)
    plt.show()

if __name__ == "__main__":
    main()