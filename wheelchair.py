class Wheelchair:
    def __init__(self, pos, direction=(1, 0)):
        self.pos = pos
        self.direction = direction
        self.goal = None

    def move(self, cell):
        self.pos = cell

    def visible_cells(self, grid, length=10, width=3):
        cells = []
        x, y = self.pos
        dx, dy = self.direction
        for i in range(1, length+1):
            cx = int(x + dx*i)
            cy = int(y + dy*i)
            for w in range(-width, width+1):
                xx = cx + dy*w
                yy = cy - dx*w
                if 0 <= xx < len(grid[0]) and 0 <= yy < len(grid):
                    cells.append((xx, yy))
        return cells

