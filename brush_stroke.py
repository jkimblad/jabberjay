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
        temp += "size_y: " + str(self.size[1]) + "\n\n"

        return temp
