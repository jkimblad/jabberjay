#!/usr/bin/env python3

import cv2
import numpy as np
import random


def read_brush(size):
    img = cv2.imread("./brushes/1.jpg", cv2.IMREAD_GRAYSCALE)

    # img = cv2.imread("./brushes/4-removebg-preview.png")
    dim = (size, size)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def show_painting(window_name, img):
    # Normalize from 0-255 to 0-1 which openCV likes =)
    img = img / 255.0
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

    # cv2.destroyAllWindows()


def paint(canvas, brush_img, brushstroke):
    pos = brushstroke.pos

    # brush_img = np.multiply(brush_img, brushstroke.color)

    # special case, brush pos outside of canvas
    # brush_img = brush_img[0:brush_img.shape[0]]
    
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

    myClr = np.copy(brush_img)
    myClr[:, :] = brushstroke.color * 255
    alpha = np.ceil(brush_img / 255.0)
    brush_img = cv2.multiply(alpha, myClr.astype(float))
    roi = cv2.multiply((1 - alpha), roi)

    roi = cv2.add(roi, brush_img)
    # roi = brush_img
    roi = np.clip(roi, 0.0, 255.0)
    canvas[pos[0]:pos[0] + brush_img.shape[0], pos[1] :pos[1] + brush_img.shape[1]] = roi.astype(np.uint8)

    return canvas


def main():
    np.random.seed(500)  # Set seed for easier debugging
    width = 500
    height = 500
    num_brushstrokes = 10
    kill_rate = 0.5
    mutation_rate = 0.05
    # load target image
    target = cv2.imread("./target1.png", cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC)
    # create painting
    canvas = np.zeros([width, height])

    # load brush
    brush_size = 50
    brush_img = read_brush(brush_size)

    population = Population(10)

    # Populate population
    # TODO: put into population __init__
    for i in range(population.size):
        sl = create_random_strokelayer(
            num_brushstrokes, width, height, brush_size)
        population.populate(sl)

    # Evolve unto next generation
    next_picture = False
    # while True:
    num_generations = 1000
    num_evolves = 20
    window_name = 'Image de Lena'
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 500, 500)
    cv2.namedWindow("target")
    cv2.resizeWindow("target", 500, 500)
    show_painting("target", target)
    for i in range(num_generations):
        for j in range(num_evolves):
            population.evolve(
                mutation_rate,
                kill_rate,
                canvas,
                brush_img,
                target)
        # Chose top-scoring stroke_layer and add it to canvas
        for stroke in population.stroke_layers[0].brush_strokes:
            canvas = paint(canvas, brush_img, stroke)

        if i % 10 == 0:
            print(population.stroke_layers[0].score)
            show_painting(window_name, canvas)


    

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

# TODO: refactor into population


def create_random_strokelayer(num_brushstrokes, width, height, brush_size):
    brushstrokes = []
    for i in range(num_brushstrokes):
        color = np.random.rand(1)[0]
        # color = 1.0
        pos = [
            np.random.randint(width - brush_size, size=1)[0],
            np.random.randint(height - brush_size, size=1)[0]
        ]
        brushstrokes.append(BrushStroke(-1, color, pos))
    return StrokeLayer(brushstrokes)


class Population:

    def __init__(self, size):
        self.stroke_layers = []
        self.size = size

    # Evolve into next generation
    # TODO: keep brush_img (and maybe target) out of population
    def evolve(self, mutation_rate, kill_rate, canvas, brush_img, target):
        self.__score_strokelayers(canvas, target, brush_img)

        # Selection phase
        # TODO: check if we should do Tournament or Roulette instead of Rank
        self.__rank(kill_rate)

        # Add offspring
        i = 0
        while len(self.stroke_layers) < self.size:
            offspring = self.__crossover(
                self.stroke_layers[i],
                self.stroke_layers[i + 1]
            )
            # Check for mutation
            rand = np.random.rand(1)[0]
            if rand <= mutation_rate:
                offspring.mutate(canvas.shape)

            self.populate(offspring)
            i += 1

    def populate(self, ls):
        self.stroke_layers.append(ls)

    def __score_strokelayers(self, canvas, target, brush_img):
        max_score = 255 * target.shape[0] * target.shape[1]
        for stroke_layer in self.stroke_layers:
            tmp_cavnas = np.copy(canvas)
            # apply stroke_layer
            for brush_stroke in stroke_layer.brush_strokes:
                tmp_cavnas = paint(tmp_cavnas, brush_img, brush_stroke)
            # check diff from target
            diff = np.subtract(target, tmp_cavnas)
            diff = np.abs(diff)
            diff = np.sum(diff)
            stroke_layer.score = max_score - diff

        def get_score(ls):
            return ls.score

        self.stroke_layers.sort(key=get_score, reverse=True)

    def __crossover(self, strokelayer_1, strokelayer_2):
        # Combine bushstrokes randomly and make children with 5 strokes each
        brush_strokes_1 = strokelayer_1.brush_strokes
        brush_strokes_2 = strokelayer_2.brush_strokes

        brush_stroke_offspring = brush_strokes_1[:int(
            len(brush_strokes_1) / 2)] + brush_strokes_2[int(len(brush_strokes_2) / 2 - 1):-1]

        return StrokeLayer(brush_stroke_offspring)

    # Selection methods
    def __rank(self, kill_rate):
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

    def mutate(self, screen_size):
        for brush_stroke in self.brush_strokes:
            # Random color
            brush_stroke.color = np.random.rand()

            # random direction, up left down right
            x_dir = random.choice([-1, 1])
            y_dir = random.choice([-1, 1])
            # random amount of change, 0-10% of screen size?
            x_factor = np.random.rand(100)[0]
            y_factor = np.random.rand(100)[0]

            brush_stroke.pos = [
                int(np.round(brush_stroke.pos[0] + x_dir * screen_size[0] * (x_factor / 100))), 
                int(np.round(brush_stroke.pos[1] + y_dir * screen_size[1] * (y_factor / 100)))
            ]



class BrushStroke:

    def __init__(self, scale_factor, color, pos):
        self.scale_factor = scale_factor
        self.color = color
        self.pos = pos


if __name__ == '__main__':
    main()
