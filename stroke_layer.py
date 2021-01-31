import numpy as np

class StrokeLayer:

    def __init__(self, bs):
        self.indx = np.random.randint(999, size=1)[0]

        self.score = 0
        self.dscore = 0
        self.brush_strokes = bs

    def __str__(self):
        temp = "SL "+str(self.indx)+" score: " + str(self.score) + " dscore: " + str(self.dscore) + "\n"
        # for brush_stroke in self.brush_strokes:
            # temp += brush_stroke.__str__()

        return temp

    def mutate(self, screen_size):
        for brush_stroke in self.brush_strokes:
            # Random color
            brush_stroke.color = np.random.rand()

            new_x_pos, new_y_pos = self.__get_new_pos(brush_stroke, screen_size)

            brush_stroke.pos = [
                int(new_x_pos),
                int(new_y_pos),
            ]

    def __get_new_pos(self, brush_stroke, screen_size):
        new_x_pos = np.random.randint(0, screen_size[0], size=1)[0]
        new_y_pos = np.random.randint(0, screen_size[1], size=1)[0]

        # # random direction, up left down right
        # x_dir = random.choice([-1, 1])
        # y_dir = random.choice([-1, 1])
        # # random amount of change, 0-10% of screen size?
        # x_factor = np.random.rand(100)[0]
        # y_factor = np.random.rand(100)[0]
        # # Calculate new x pos randomly
        # new_x_pos = np.round(brush_stroke.pos[0] + x_dir * screen_size[0] * (x_factor / 100))
        # # Clip to ensure it's within screen dimensions
        new_x_pos = np.clip(new_x_pos, 1 - brush_stroke.size, screen_size[0] - 1)

        # # Calculate new y pos randomly
        # new_y_pos = np.round(brush_stroke.pos[1] + y_dir * screen_size[1] * (y_factor / 100))
        # # Clip to ensure it's within screen dimensions
        new_y_pos = np.clip(new_y_pos, 1 - brush_stroke.size, screen_size[1] - 1)

        return new_x_pos, new_y_pos
