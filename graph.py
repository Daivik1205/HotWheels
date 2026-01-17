import heapq
from math import inf


def heuristic(a, b, coords):
    ax, ay = coords[a]
    bx, by = coords[b]
    return abs(ax - bx) + abs(ay - by)


class PathPlanner:
    def __init__(self, graph, coords):
        self.graph = graph
        self.coords = coords

    def a_star(self, start, goal):
        pq = [(0, start)]
        g = {start: 0}
        parent = {start: None}

        while pq:
            _, u = heapq.heappop(pq)
            if u == goal:
                break

            for v, cost in self.graph[u]:
                ng = g[u] + cost
                if ng < g.get(v, inf):
                    g[v] = ng
                    f = ng + heuristic(v, goal, self.coords)
                    parent[v] = u
                    heapq.heappush(pq, (f, v))

        path = []
        cur = goal
        while cur:
            path.append(cur)
            cur = parent.get(cur)
        return list(reversed(path))

