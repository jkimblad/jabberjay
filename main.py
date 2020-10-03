#!/usr/bin/env python3

import cv2
import numpy as np


def read_brush(size):
    img = cv2.imread("./brushes/1.jpg", cv2.IMREAD_GRAYSCALE)

    # img = cv2.imread("./brushes/4-removebg-preview.png")
    dim = (size, size)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def show_painting(img):
    # Normalize from 0-255 to 0-1 which openCV likes =)
    img = img / 255.0
    cv2.imshow("image", img)
    cv2.waitKey()


def paint(img, brush, pos, brightness):
    if img.ndim != brush.ndim:
        raise Exception("Mismatch in img and brush dimensions")

    brush = np.multiply(brush, brightness)

    roi = img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]]
    roi = cv2.add(roi, brush)
    roi = np.clip(roi, 0.0, 255.0)
    img[pos[0]:pos[0] + brush.shape[0], pos[1]
        :pos[1] + brush.shape[1]] = roi.astype(np.uint8)
    print(img)

    return img

def paint(img, brush_img, brushstroke):
    pos = brushstroke.pos
    brush_img = np.multiply(brush_img, brushstroke.brightness)

    roi = img[pos[0]:pos[0] + brush_img.shape[0], pos[1]:pos[1] + brush_img.shape[1]]
    roi = cv2.add(roi, brush_img)
    roi = np.clip(roi, 0.0, 255.0)
    img[pos[0]:pos[0] + brush_img.shape[0], pos[1]
        :pos[1] + brush_img.shape[1]] = roi.astype(np.uint8)

    return img


def main():
    np.random.seed(50) # Set seed for easier debugging
    width = 500
    height = 500
    # load target image
    target = cv2.imread("./target.jpg", cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC)
    # create painting
    img = np.zeros([width, height])

    # load brush
    brush_size = 50
    brush = read_brush(brush_size)
    print(brush.shape)

    population = Population(10)

    for i in range(population.population_size):
        bs = create_random_brushstroke(width, height, brush_size)
        population.populate(bs)

    population.score_brushstrokes(img, target, brush)

    # # Draw brushstrokes
    # for x in range(100):

    #     # get random pos
    #     pos = (
    #         np.random.randint(width - brush_size, size=1)[0],
    #         np.random.randint(height - brush_size, size=1)[0]
    #     )

    #     # print(pos)
    #     # stroke brush on painting
    #     img = paint(img, brush, pos, np.random.rand(1)[0])

    # # show painting
    # show_painting(img)

def create_random_brushstroke(width, height, brush_size):
    brightness = np.random.rand(1)[0]
    pos = (
            np.random.randint(width - brush_size, size=1)[0],
            np.random.randint(height - brush_size, size=1)[0]
        )
    return BrushStroke(-1, brightness, pos)

class Population:

    def __init__(self, population_size):
        self.brushstrokes = []
        self.population_size = population_size

    def populate(self, brushstroke):
        self.brushstrokes.append(brushstroke)

    def score_brushstrokes(self, canvas, target, brush_img):
        max_score = 255 * target.shape[0] * target.shape[1]
        for brushstroke in self.brushstrokes:
            tmp_cavnas = np.copy(canvas)
            # apply brushstroke
            tmp_cavnas = paint(tmp_cavnas, brush_img, brushstroke)
            # check diff from target
            diff = np.subtract(target, tmp_cavnas)
            diff = np.abs(diff)
            diff = np.sum(diff)
            brushstroke.score = max_score - diff

        def get_score(bs):
            return bs.score

        self.brushstrokes.sort(key=get_score, reverse=True)


    def crossover(self, brushstroke_1, brushstroke_2):
        # Queue marvin gaye- lets get it on
        pass

    def mutate(self, brushstroke):
        pass


class BrushStroke:

    def __init__(self, scale_factor, brightness, pos):
        self.scale_factor = scale_factor
        self.brightness = brightness
        self.pos = pos
        self.score = 0


if __name__ == '__main__':
    main()
