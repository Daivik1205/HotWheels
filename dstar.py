import heapq
import math

INF = float("inf")

# 8-neighborhood moves (dx, dy, base_cost)
MOVES = [
    (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
    (1, 1, 1.41421356237), (-1, 1, 1.41421356237),
    (1, -1, 1.41421356237), (-1, -1, 1.41421356237),
]


class DStarLite:
    """
    Correct D* Lite using grid[y, x] with shape (H, W).
    grid[y, x] = cost to enter cell (x,y)
    blocked => INF
    """

    def __init__(self, start, goal, grid):
        self.grid = grid  # [H,W]
        self.start = start
        self.goal = goal

        self.last = start          # used for km update
        self.prev_pos = None       # TRUE previous position (for jitter kill)

        self.km = 0.0

        self.g = {}
        self.rhs = {}
        self.U = []

        H, W = self.grid.shape
        for y in range(H):
            for x in range(W):
                self.g[(x, y)] = INF
                self.rhs[(x, y)] = INF

        self.rhs[self.goal] = 0.0
        heapq.heappush(self.U, (self.key(self.goal), self.goal))

    def in_bounds(self, s):
        x, y = s
        H, W = self.grid.shape
        return 0 <= x < W and 0 <= y < H

    def traversable(self, s):
        x, y = s
        c = float(self.grid[y, x])
        return (c > 0.0) and (not math.isinf(c))

    def h(self, a, b):
        # Manhattan is stable
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def key(self, s):
        m = min(self.g[s], self.rhs[s])
        return (m + self.h(self.start, s) + self.km, m)

    def cost(self, a, b):
        """Move cost a->b = base * enter_cost(b)."""
        bx, by = b
        cell_cost = float(self.grid[by, bx])
        if cell_cost <= 0.0 or math.isinf(cell_cost):
            return INF

        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        if dx + dy == 1:
            base = 1.0
        elif dx == 1 and dy == 1:
            base = 1.41421356237
        else:
            return INF

        return base * cell_cost

    def successors(self, s):
        sx, sy = s
        for dx, dy, _ in MOVES:
            nx, ny = sx + dx, sy + dy
            ns = (nx, ny)

            if not self.in_bounds(ns):
                continue
            if not self.traversable(ns):
                continue

            # FORBID diagonal corner cutting
            if dx != 0 and dy != 0:
                if not (
                    self.traversable((sx + dx, sy)) and
                    self.traversable((sx, sy + dy))
                ):
                    continue

            yield ns

    def predecessors(self, s):
        return self.successors(s)

    def update_vertex(self, u):
        if u != self.goal:
            best = INF
            for s in self.successors(u):
                cand = self.g[s] + self.cost(u, s)
                if cand < best:
                    best = cand
            self.rhs[u] = best

        heapq.heappush(self.U, (self.key(u), u))

    def _top_key(self):
        while self.U:
            k, u = self.U[0]
            if k != self.key(u):
                heapq.heappop(self.U)
                continue
            return k
        return (INF, INF)

    def compute(self, max_steps=None):
        steps = 0
        while True:
            topk = self._top_key()
            if not ((topk < self.key(self.start)) or (self.rhs[self.start] != self.g[self.start])):
                break
            if not self.U:
                break

            k_old, u = heapq.heappop(self.U)
            if k_old != self.key(u):
                continue

            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for p in self.predecessors(u):
                    self.update_vertex(p)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for p in self.predecessors(u):
                    self.update_vertex(p)

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        return steps

    def update_cell_cost(self, cell, new_cost):
        x, y = cell
        self.grid[y, x] = float(new_cost)
        self.update_vertex(cell)
        for p in self.predecessors(cell):
            self.update_vertex(p)

    def update_obstacle(self, cell):
        self.update_cell_cost(cell, INF)

    def move_start(self, new_start):
        if new_start == self.start:
            return
        self.km += self.h(self.last, new_start)
        self.last = new_start
        self.start = new_start

    def next(self, compute_budget=800):
        self.compute(max_steps=compute_budget)

        succs = list(self.successors(self.start))
        if not succs:
            return None

        # REAL previous (used only to stop jitter)
        prev = self.prev_pos

        # Direction we were traveling
        prev_dir = None
        if prev is not None:
            prev_dir = (self.start[0] - prev[0], self.start[1] - prev[1])

        best = None
        best_rank = (INF, INF, INF, INF)

        for s in succs:
            step_cost = self.cost(self.start, s)
            if math.isinf(step_cost):
                continue

            # primary D* metric
            val = self.g[s] + step_cost
            if math.isinf(val):
                continue

            # jitter killers
            backtrack = 1 if (prev is not None and s == prev) else 0
            h_goal = self.h(s, self.goal)

            # prefer going straight if possible
            step_dir = (s[0] - self.start[0], s[1] - self.start[1])
            turn = 0
            if prev_dir is not None and step_dir != prev_dir:
                turn = 1

            # RANK ORDER is important:
            # 1) best D* value
            # 2) avoid going back
            # 3) avoid turning
            # 4) closer to goal
            rank = (val, backtrack, turn, h_goal)

            if rank < best_rank:
                best_rank = rank
                best = s

        if best is None:
            return None

        # update prev_pos BEFORE moving start
        self.move_start(best)
        return best

