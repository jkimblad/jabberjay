#!/usr/bin/env python3

import cv2
import numpy as np
import random

from population import Population
# from brush_stroke import BrushStroke
# from stroke_layer import StrokeLayer

DEBUG = 1

def read_brush(size):
    img = cv2.imread("./brushes/1.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    return img

def show_painting(window_name, img):
    # Normalize from 0-255 to 0-1 which openCV likes =)
    img = img / 255.0
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

def paint(canvas, brush_img, brushstroke):
    pos = brushstroke.pos

    #resize the brush
    brush_img = cv2.resize(brush_img, brushstroke.size, interpolation = cv2.INTER_CUBIC)

    if pos[0] < 0:
        brush_img = brush_img[0:brush_img.shape[0] + pos[0], :]
        pos[0] = 0
    if pos[1] < 0:
        brush_img = brush_img[:, 0:brush_img.shape[1] + pos[1]]
        pos[1] = 0

    roi = canvas[pos[0]:pos[0] + brush_img.shape[0],
              pos[1]:pos[1] + brush_img.shape[1]]

    # Crop brush_img to the same size of roi, this occurs if pos is outside of canvas
    brush_img = brush_img[:roi.shape[0], :roi.shape[1]]
    # rotate, credit to anopara for this code. Not sure how it works exactly
    rows, cols = brush_img.shape
    M = cv2.getRotationMatrix2D( (cols/2, rows/2), brushstroke.rot, 1)
    brush_img = cv2.warpAffine(brush_img, M, (cols, rows))

    myClr = np.copy(brush_img)
    myClr[:, :] = brushstroke.color * 255
    alpha = np.ceil(brush_img / 255.0)
    brush_img = cv2.multiply(alpha, myClr.astype(float))
    roi = cv2.multiply((1 - alpha), roi)

    roi = cv2.add(roi, brush_img)
    roi = np.clip(roi, 0.0, 255.0)

    canvas[pos[0]:pos[0] + brush_img.shape[0], pos[1] :pos[1] + brush_img.shape[1]] = roi.astype(np.uint8)

    return canvas

def main():
    np.random.seed(500)  # Set seed for easier debugging
    width = 500
    height = 500
    num_brushstrokes = 4
    kill_rate = 0.5
    mutation_rate = 0.1
    # load target image
    target = cv2.imread("./photos/mona.jpg", cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC)
    # create painting
    canvas = np.zeros([width, height])
    # load brush
    brush_max_size = (80, 50)
    brush_img = read_brush(brush_max_size)

    # Create and populate population
    population = Population(20, num_brushstrokes, width, height, brush_max_size)

    # Evolve unto next generation
    next_picture = False
    # while True:
    num_generations = 10000
    num_evolves = 3
    window_name = 'Image de Lena'
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 500, 500)
    cv2.moveWindow(window_name, 600, 100)
    cv2.namedWindow("target")
    cv2.resizeWindow("target", 500, 500)
    cv2.moveWindow("target", 100, 100)
    if (DEBUG):
        cv2.namedWindow("debug")
        cv2.resizeWindow("debug", 500, 500)
        cv2.moveWindow("debug", 1100, 100)

    show_painting("target", target)
    for i in range(num_generations):
        for j in range(num_evolves):
            population.evolve(
                mutation_rate,
                kill_rate,
                canvas,
                brush_img,
                target,
                paint)
        # Chose top-scoring stroke_layer and add it to canvas
        for stroke in population.stroke_layers[0].brush_strokes:
            canvas = paint(canvas, brush_img, stroke)

        debug_canvas = np.array([0])
        # Draw each StrokeLayer in the population in a new window after num_generations
        if (DEBUG):
            debug_canvas = np.zeros([width, height])
            for stroke_layer in population.stroke_layers:
                for stroke in stroke_layer.brush_strokes:
                    debug_canvas = paint(debug_canvas, brush_img, stroke)

        if i % 1 == 0:
            if(DEBUG):
                print("0:", population.stroke_layers[0])
                print("1:", population.stroke_layers[1])
                print("2:", population.stroke_layers[2])
                show_painting("debug", debug_canvas)

        show_painting(window_name, canvas)

    # Save image
    cv2.imwrite("./photos/painted.png", canvas)

if __name__ == '__main__':
    main()
