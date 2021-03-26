import numpy as np
import sys
from brush_stroke import BrushStroke, create_random_brushstroke
from stroke_layer import StrokeLayer

class Population:

    def __init__(self, size, num_brushstrokes, width, height, brush_max_size):
        self.stroke_layers = []
        self.size = size
        self.crossover_method = "random" # TODO: pass this in parameter
        # TODO pass this as parameters in evolve instead of storing in class.. maybe
        self.width = width
        self.height = height
        self.brush_size = brush_max_size

        # Populate the population
        for i in range(size):
            sl = create_random_strokelayer(
                num_brushstrokes, width, height, brush_max_size)
            self.__populate(sl)

    # Evolve into next generation
    # TODO: keep brush_img (and maybe target) out of population
    def evolve(self, mutation_rate, kill_rate, canvas, brush_img, target, paint):
        self.__score_strokelayers(canvas, target, brush_img, paint)

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

            self.__populate(offspring)
            i += 1

    def __populate(self, ls):
        self.stroke_layers.append(ls)

    def __score_strokelayers(self, canvas, target, brush_img, paint):
        max_score = 255 * target.shape[0] * target.shape[1]
        diff = np.subtract(target, canvas)
        diff = np.abs(diff)
        diff = np.sum(diff)
        canvas_score = max_score - diff
        for stroke_layer in self.stroke_layers:
            tmp_canvas = np.copy(canvas)
            # apply stroke_layer
            for brush_stroke in stroke_layer.brush_strokes:
                tmp_canvas = paint(tmp_canvas, brush_img, brush_stroke)
            # check diff from target
            diff = np.subtract(target, tmp_canvas)
            diff = np.abs(diff)
            diff = np.sum(diff)
            stroke_layer.score = max_score - diff
            stroke_layer.dscore = stroke_layer.score - canvas_score

        def get_score(ls):
            return ls.score

        self.stroke_layers.sort(key=get_score, reverse=True)

    def __crossover(self, strokelayer_1, strokelayer_2):
        # Combine bushstrokes randomly and make children with 5 strokes each
        brush_strokes_1 = strokelayer_1.brush_strokes
        brush_strokes_2 = strokelayer_2.brush_strokes

        brush_stroke_offspring = []
        # "I suspect that this method is not so good, easily get stuck in local minima" - JH
        if self.crossover_method == "average":
            for i in range(len(brush_strokes_1)):
                # Take average all from first and second
                new_color = (brush_strokes_1[i].color + brush_strokes_2[i].color) / 2
                new_x_pos = (brush_strokes_1[i].pos[0] + brush_strokes_2[i].pos[0]) / 2
                new_y_pos = (brush_strokes_1[i].pos[1] + brush_strokes_2[i].pos[1]) / 2
                # new_size = (brush_strokes_1[i].size + brush_strokes_2[i].size) / 2
                new_size = ((brush_strokes_1[i].size[0] + brush_strokes_2[i].size[0]) / 2, (brush_strokes_1[i].size[1] + brush_strokes_2[i].size[1]) / 2)
                new_rot =   (brush_strokes_1[i].rot + brush_strokes_2[i].rot) / 2
                brush_stroke_offspring.append(BrushStroke(new_color, [int(round(new_x_pos)), int(round(new_y_pos))], new_size, new_rot))
        elif self.crossover_method == "random":
            for i in range(len(brush_strokes_1)):
                brush_stroke_offspring.append(create_random_brushstroke(self.width, self.height, self.brush_size))
        else:
            sys.exit("Invalid crossover_method, expecting average|random")



        return StrokeLayer(brush_stroke_offspring)

    # Selection methods
    def __rank(self, kill_rate):
        pop_size = len(self.stroke_layers)

        # Check that the kill_rate will leave at least 2 pop
        new_pop_size = int(pop_size * (1 - kill_rate))
        if new_pop_size <= 2:
            raise Exception("Kill Ratio is too agressive")

        self.stroke_layers = self.stroke_layers[:new_pop_size]


# TODO: refactor into population
def create_random_strokelayer(num_brushstrokes, width, height, brush_size):
    brushstrokes = []
    for i in range(num_brushstrokes):
        brushstrokes.append(create_random_brushstroke(width, height, brush_size))
    return StrokeLayer(brushstrokes)
