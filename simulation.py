from dstar import DStarLite
from viewer import Viewer
from wheelchair import Wheelchair


class Simulation:
    def __init__(self, grid):
        self.grid = grid
        self.viewer = Viewer(len(grid[0]), len(grid))
        self.wc = Wheelchair((5,5))
        self.local = None
        self.global_path = []

    def run(self, global_path):
        self.global_path = global_path
        self.wc.goal = global_path[-1]

        running = True
        while running:
            running = self.viewer.events()

            fov = self.wc.visible_cells(self.grid)

            if self.local is None:
                self.local = DStarLite(self.wc.pos, self.global_path[1], self._copy(self.grid))
                self.local.compute()

            step = self.local.next()
            self.wc.move(step)

            self.viewer.draw(self.grid, self.wc, fov, global_path)

    def _copy(self, grid):
        return [row[:] for row in grid]

