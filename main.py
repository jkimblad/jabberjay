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

    # resize the brush
    brush_img = cv2.resize(
        brush_img,
        brushstroke.size,
        interpolation=cv2.INTER_CUBIC)

    if pos[0] < 0:
        brush_img = brush_img[0:brush_img.shape[0] + pos[0], :]
        pos[0] = 0
    if pos[1] < 0:
        brush_img = brush_img[:, 0:brush_img.shape[1] + pos[1]]
        pos[1] = 0

    roi = canvas[pos[0]:pos[0] + brush_img.shape[0],
                 pos[1]:pos[1] + brush_img.shape[1]]

    # Crop brush_img to the same size of roi, this occurs if pos is outside of
    # canvas
    brush_img = brush_img[:roi.shape[0], :roi.shape[1]]
    # rotate, credit to anopara for this code. Not sure how it works exactly
    rows, cols = brush_img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), brushstroke.rot, 1)
    brush_img = cv2.warpAffine(brush_img, M, (cols, rows))

    myClr = np.copy(brush_img)
    myClr[:, :] = brushstroke.color * 255
    alpha = np.ceil(brush_img / 255.0)
    brush_img = cv2.multiply(alpha, myClr.astype(float))
    roi = cv2.multiply((1 - alpha), roi)

    roi = cv2.add(roi, brush_img)
    roi = np.clip(roi, 0.0, 255.0)

    canvas[pos[0]:pos[0] + brush_img.shape[0], pos[1]
        :pos[1] + brush_img.shape[1]] = roi.astype(np.uint8)

    return canvas


def main():
    np.random.seed(500)  # Set seed for easier debugging
    width = 500
    height = 500

    num_brushstrokes = 4
    kill_rate = 0.5
    mutation_rate = 0.1

    # create painting
    canvas = np.zeros([width, height])
    # load brush
    brush_max_size = (80, 50)
    brush_img = read_brush(brush_max_size)

    # Create and populate population
    population = Population(
        20,
        num_brushstrokes,
        width,
        height,
        brush_max_size)

    num_evolves = 3

    window_name = 'Image de Lena'
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 500, 500)
    cv2.moveWindow(window_name, 600, 100)

    cv2.namedWindow("target")
    cv2.resizeWindow("target", 500, 500)
    cv2.moveWindow("target", 100, 100)

    cam = cv2.VideoCapture(0)

    # create painting
    canvas = np.zeros([width, height])

    while True:
        # Read image input
        ret, frame = cam.read()
        if not ret:
            print("failed to grab target")
            break
        target = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target = cv2.flip(target, 1)
        target = cv2.resize(target, (width, height),
                            interpolation=cv2.INTER_CUBIC)

        show_painting("target", target)

        # run algo on target image
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

        show_painting(window_name, canvas)


if __name__ == '__main__':
    main()
