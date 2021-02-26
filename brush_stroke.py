class BrushStroke:

    def __init__(self, color, pos, size):
        self.color = color
        self.pos = pos
        self.size = size

    def __str__(self):
        temp = "color: " + str(self.color) + "\n"
        temp += "pos_x: " + str(self.pos[0]) + "\n"
        temp += "pos_y: " + str(self.pos[1]) + "\n"
        temp += "size: " + str(self.size) + "\n\n"

        return temp
