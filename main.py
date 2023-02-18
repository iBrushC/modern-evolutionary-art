import numpy as np
import cv2
import os

from imageops import *

def main():
    skull = cv2.imread("images/skull.jpg")
    skull = cv2.cvtColor(skull, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    square_skull = fit_image_to_square(image=skull, n=196)
    
    line_count = 128
    line_canvas = np.ones(shape=(196, 196), dtype=np.uint8) * 255
    lines = np.random.rand(line_count, 2, 2) * 196
    lines = lines.astype(np.int32)

    for line in lines:
        line_canvas = cv2.line(line_canvas, line[0], line[1], 0, 1)

    blurred_canvas = cv2.blur(line_canvas, ksize=(3, 3))
    cv2.imshow("random", blurred_canvas)
    cv2.waitKey()
    

if __name__ == "__main__":
    main()