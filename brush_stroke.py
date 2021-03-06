import numpy as np

class BrushStroke:

    def __init__(self, color, pos, size, rot):
        self.color = color
        self.pos = pos
        self.size = size
        self.rot = rot

    def __str__(self):
        temp = "color: " + str(self.color) + "\n"
        temp += "pos_x: " + str(self.pos[0]) + "\n"
        temp += "pos_y: " + str(self.pos[1]) + "\n"
        temp += "size_x: " + str(self.size[0]) + "\n"
        temp += "size_y: " + str(self.size[1]) + "\n"
        temp += "rot: " + str(self.rot) + "\n\n"

        return temp


def create_random_brushstroke(width, height, brush_size):
    color = np.random.rand(1)[0]
    pos = [
        np.random.randint(width - brush_size[0], size=1)[0],
        np.random.randint(height - brush_size[1], size=1)[0]
    ]
    rot = np.random.randint(-180, 180, size=1)[0]
    b_size = (np.random.randint(1, brush_size[0], size=1)[0], np.random.randint(1, brush_size[1], size=1)[0])
    return BrushStroke(color, pos, b_size, rot)
