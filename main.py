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
    img[pos[0]:pos[0] + brush.shape[0], pos[1]:pos[1] + brush.shape[1]] = roi.astype(np.uint8)
    print(img)

    return img


def paint(img, brush_img, brushstroke):
    pos = brushstroke.pos
    brush_img = np.multiply(brush_img, brushstroke.brightness)

    roi = img[pos[0]:pos[0] + brush_img.shape[0],
              pos[1]:pos[1] + brush_img.shape[1]]
    roi = cv2.add(roi, brush_img)
    roi = np.clip(roi, 0.0, 255.0)
    img[pos[0]:pos[0] + brush_img.shape[0], pos[1]:pos[1] + brush_img.shape[1]] = roi.astype(np.uint8)

    return img


def main():
    np.random.seed(50)  # Set seed for easier debugging
    width = 500
    height = 500
    num_brushstrokes = 10
    kill_rate = 0.5
    mutation_rate = 0.05
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

    for i in range(population.size):
        sl = create_random_strokelayer(
            num_brushstrokes, width, height, brush_size)
        population.populate(sl)

    population.score_strokelayers(img, target, brush)

    # Selection phase
    # TODO: check if we should do Tournament or Roulette instead of Rank
    population.rank(kill_rate)

    # Add offspring
    i = 0
    while len(population.stroke_layers) < population.size:
        offspring = population.crossover(
            population.stroke_layers[i],
            population.stroke_layers[i + 1]
        )
        # Check for mutation
        rand = np.random.rand(1)[0]
        if rand <= mutation_rate:
            offspring.mutate(mutation_rate)

        population.populate(offspring)
        i += 1

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


def create_random_strokelayer(num_brushstrokes, width, height, brush_size):
    brushstrokes = []
    for i in range(num_brushstrokes):
        brightness = np.random.rand(1)[0]
        pos = (
            np.random.randint(width - brush_size, size=1)[0],
            np.random.randint(height - brush_size, size=1)[0]
        )
        brushstrokes.append(BrushStroke(-1, brightness, pos))
    return StrokeLayer(brushstrokes)


class Population:

    def __init__(self, size):
        self.stroke_layers = []
        self.size = size

    def populate(self, ls):
        self.stroke_layers.append(ls)

    def score_strokelayers(self, canvas, target, brush_img):
        max_score = 255 * target.shape[0] * target.shape[1]
        for stroke_layer in self.stroke_layers:
            tmp_cavnas = np.copy(canvas)
            # apply stroke_layer
            for brushstroke in stroke_layer.brush_strokes:
                tmp_cavnas = paint(tmp_cavnas, brush_img, brushstroke)
            # check diff from target
            diff = np.subtract(target, tmp_cavnas)
            diff = np.abs(diff)
            diff = np.sum(diff)
            stroke_layer.score = max_score - diff

        def get_score(ls):
            return ls.score

        self.stroke_layers.sort(key=get_score, reverse=True)

    def crossover(self, strokelayer_1, strokelayer_2):
        # Combine bushstrokes randomly and make children with 5 strokes each
        brush_strokes_1 = strokelayer_1.brush_strokes
        brush_strokes_2 = strokelayer_2.brush_strokes

        brush_stroke_offspring = [
            brush_strokes_1[1:int(len(brush_strokes_1) / 2)],
            brush_strokes_2[int(len(brush_strokes_2) / 2):-1]
        ]

        return StrokeLayer(brush_stroke_offspring)

    # Selection methods
    def rank(self, kill_rate):
        pop_size = len(self.stroke_layers)

        # Check that the kill_rate will leave at least 2 pop
        new_pop_size = int(pop_size * kill_rate)
        if new_pop_size <= 2:
            raise Exception("Kill Ratio is too agressive")

        self.stroke_layers = self.stroke_layers[:new_pop_size]


class StrokeLayer:

    def __init__(self, bs):
        self.score = 0
        self.brush_strokes = bs

    def mutate(self, stroke_layer):
        pass


class BrushStroke:

    def __init__(self, scale_factor, brightness, pos):
        self.scale_factor = scale_factor
        self.brightness = brightness
        self.pos = pos


if __name__ == '__main__':
    main()
