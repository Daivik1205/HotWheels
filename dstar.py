import heapq
from math import inf


def h(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


class DStarLite:
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.U = []
        self.g = {}
        self.rhs = {}

        for y in range(len(grid)):
            for x in range(len(grid[0])):
                self.g[(x, y)] = inf
                self.rhs[(x, y)] = inf

        self.rhs[goal] = 0
        self._push(goal, h(start, goal))

    def _push(self, node, p):
        heapq.heappush(self.U, (p, node))

    def _succ(self, n):
        x, y = n
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= nx < len(self.grid[0]) and 0 <= ny < len(self.grid):
                yield (nx, ny)

    def _update(self, n):
        if n != self.goal:
            self.rhs[n] = min(self.g[s] + 1 for s in self._succ(n))
        self._push(n, min(self.g[n], self.rhs[n]) + h(self.start, n))

    def compute(self):
        while self.U and (
            self.U[0][0] < self.g[self.start] or
            self.rhs[self.start] != self.g[self.start]
        ):
            _, u = heapq.heappop(self.U)
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = inf
                self._update(u)
            for s in self._succ(u):
                if self.grid[s[1]][s[0]] == 1:
                    continue
                self._update(s)

    def update_obstacle(self, c):
        x, y = c
        self.grid[y][x] = 1
        self._update(c)

    def next(self):
        from math import inf
        best = None
        best_cost = inf
        for s in self._succ(self.start):
            c = self.g.get(s, inf)
            if c < best_cost:
                best = s
                best_cost = c
        self.start = best
        return best

