import os
import pickle
import numpy as np
import heapq
import math
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from map_creator import MapPixel, FuncID
from dstar import DStarLite


@dataclass
class TeleportInfo:
    position: Tuple[int, int]
    func_type: int
    destinations: List[str]
    cost: int
    dept_id: int
    floor: int


@dataclass
class MapInfo:
    map_id: str
    width: int
    height: int
    map_data: np.ndarray
    teleports: Dict[str, List[TeleportInfo]]


class NavigationGraph:
    def __init__(self):
        self.maps: Dict[str, MapInfo] = {}
        self.graph: Dict[str, Set[str]] = {}

    def load_maps(self, maps_directory: str):
        if not os.path.exists(maps_directory):
            return
        for f in os.listdir(maps_directory):
            if f.endswith(".bin"):
                self._load_map(os.path.join(maps_directory, f))
        self._build_graph()

    def _load_map(self, filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            mid = data.get("map_id", os.path.basename(filepath)[:-4])
            w, h = data["width"], data["height"]

            pixels = np.array([[MapPixel.from_tuple(p) for p in row] for row in data["map_data"]], dtype=object)
            self.maps[mid] = MapInfo(mid, w, h, pixels, self._find_teleports(pixels, w, h))
        except:
            pass

    def _find_teleports(self, pixels, w, h):
        tps = {}
        for y in range(h):
            for x in range(w):
                p = pixels[y][x]
                if p.func_id in [FuncID.DOOR, FuncID.ELEVATOR, FuncID.RAMP, FuncID.STAIR] and p.identifier:
                    info = TeleportInfo((x, y), p.func_id, p.identifier, p.cost, p.dept_id, p.floor)
                    for d in p.identifier:
                        tps.setdefault(d, []).append(info)
        return tps

    def _build_graph(self):
        self.graph = {m: set() for m in self.maps}
        for m, info in self.maps.items():
            for d in info.teleports:
                if d in self.maps:
                    self.graph[m].add(d)
                    self.graph[d].add(m)

    def find_map_path(self, start, end):
        if start == end:
            return [start]
        q = [(start, [start])]
        vis = {start}
        while q:
            u, path = q.pop(0)
            for v in self.graph.get(u, []):
                if v not in vis:
                    if v == end:
                        return path + [v]
                    vis.add(v)
                    q.append((v, path + [v]))
        return None

    def get_teleport_to(self, curr, target):
        t = self.maps[curr].teleports.get(target, [])
        return t[0].position if t else None

    def get_entry_teleport_from(self, current_map: str, next_map: str, goal_hint=None):
        next_info = self.maps.get(next_map)
        if not next_info:
            return None
        candidates = next_info.teleports.get(current_map, [])
        if not candidates:
            return None
        if goal_hint is None:
            return candidates[0].position
        return min(
            candidates,
            key=lambda tp: abs(tp.position[0] - goal_hint[0]) + abs(tp.position[1] - goal_hint[1])
        ).position


class DynamicLocalPlanner:
    """
    D* Lite local planner with limited sight + strong wall avoidance.

    Key behavior:
    - EMPTY is expensive (200)
    - WALKABLE is cheap (1)
    - cells near obstacles get huge extra penalty (clearance_cost)
    """

    def __init__(self, map_info, start, goal, initial_visible_cells=None):
        self.map = map_info
        self.start = start
        self.goal = goal

        self.gt_costs = self._build_ground_truth_cost_grid()
        self.clearance_cost = self._build_clearance_cost(self.gt_costs)

        # Unknown/unseen cells default to expensive (assume "empty")
        self.known_costs = np.ones_like(self.gt_costs, dtype=np.float32) * 5.0

        self._sanitize_start_goal()

        self.planner = DStarLite(self.start, self.goal, self.known_costs)
        
        if initial_visible_cells is not None:
            self.sense_and_update(initial_visible_cells)
            self.planner.compute(max_steps=20000)
        else:
            self.planner.compute(max_steps=20000)

        sx, sy = self.start
        gx, gy = self.goal
        print("gt start cost", self.gt_costs[sy, sx], "gt goal cost", self.gt_costs[gy, gx])

    def _build_ground_truth_cost_grid(self):
        w, h = self.map.width, self.map.height
        grid = np.ones((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                p = self.map.map_data[y][x]
                if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
                    grid[y, x] = np.inf
                else:
                    # only used as "base", real live cost comes from map_data anyway
                    grid[y, x] = max(1.0, float(p.cost))
        return grid

    def _build_clearance_cost(self, gt):
        """
        Distance-to-obstacle -> penalty.
        Fixes:
        - inflation is applied BEFORE BFS seed + BFS uses inflated obstacles
        """
        H, W = gt.shape

        # Inflate obstacles (radius 1)
        inflated = gt.copy()
        for y in range(H):
            for x in range(W):
                if math.isinf(float(gt[y, x])):
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                inflated[ny, nx] = np.inf

        # BFS distance transform FROM inflated obstacles
        dist = np.full((H, W), 10**9, dtype=np.int32)
        q = []

        for y in range(H):
            for x in range(W):
                if math.isinf(float(inflated[y, x])):
                    dist[y, x] = 0
                    q.append((x, y))

        head = 0
        while head < len(q):
            x, y = q[head]
            head += 1
            d = dist[y, x] + 1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if d < dist[ny, nx]:
                        dist[ny, nx] = d
                        q.append((nx, ny))

        # ===== VERY STRONG WALL PENALTY =====
        # "near" controls how wide the bad zone is
        near = 6

        # base_penalty controls how expensive near-wall becomes
        base_penalty = 500.0

        penalty = np.zeros((H, W), dtype=np.float32)

        for y in range(H):
            for x in range(W):
                if math.isinf(float(inflated[y, x])):
                    penalty[y, x] = np.inf
                    continue

                d = int(dist[y, x])

                if d <= near:
                    # explosive penalty near walls
                    penalty[y, x] = base_penalty * float((near - d + 1) ** 2)
                else:
                    penalty[y, x] = 0.0

        return penalty

    def _sanitize_start_goal(self):
        def free(cell):
            x, y = cell
            return not math.isinf(float(self.gt_costs[y, x]))

        if not free(self.start) or not free(self.goal):
            return

    def _true_cost_live(self, x, y):
        p = self.map.map_data[y][x]
        if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
            return np.inf

        if p.func_id == FuncID.EMPTY:
            base = 200.0
        else:
            base = max(1.0, float(p.cost))

        return base + float(self.clearance_cost[y, x])


    def sense_and_update(self, visible_cells):
        for (x, y) in visible_cells:
            true_cost = self._true_cost_live(x, y)

            known = float(self.known_costs[y, x])
            changed = (
                (math.isinf(true_cost) and not math.isinf(known))
                or (not math.isinf(true_cost) and (math.isinf(known) or abs(true_cost - known) > 1e-6))
            )

            if changed:
                self.known_costs[y, x] = true_cost
                self.planner.update_cell_cost((x, y), true_cost)

    def step(self, compute_budget=2000):
        """
        Jitter-free movement by path caching.

        Instead of choosing just the next neighbor each frame (which jitters),
        we extract a short path from the current g-values and follow it.

        Replans only if:
        - cache is empty
        - next cached step is no longer valid
        - goal changed (rare)
        """

        # init cache
        if not hasattr(self, "_path_cache"):
            self._path_cache = []

        # helper: legal move according to D* Lite
        def is_valid_step(curr, nxt):
            for s in self.planner.successors(curr):
                if s == nxt:
                    return True
            return False

        # if we have a cached step, use it
        if self._path_cache:
            nxt = self._path_cache[0]
            if is_valid_step(self.start, nxt):
                self._path_cache.pop(0)
                self.planner.move_start(nxt)
                self.start = nxt
                return nxt
            else:
                # cached path became invalid -> drop it
                self._path_cache = []

        # compute more before extracting path
        self.planner.compute(max_steps=compute_budget)

        # extract a path by repeatedly selecting best successor
        path = []
        cur = self.start
        prev = None

        for _ in range(100):  # cache length (tune 10..60)
            if cur == self.goal:
                break

            best = None
            best_rank = (10**18, 10**18, 10**18)

            for s in self.planner.successors(cur):
                # normal D* successor scoring
                move_cost = self.planner.cost(cur, s)
                if math.isinf(move_cost):
                    continue

                val = self.planner.g[s] + move_cost
                if math.isinf(val):
                    continue

                # strong anti-jitter tie break:
                # 1) val
                # 2) avoid backtracking
                # 3) prefer straight toward goal
                backtrack = 1 if (prev is not None and s == prev) else 0
                hgoal = abs(s[0] - self.goal[0]) + abs(s[1] - self.goal[1])

                rank = (val, backtrack, hgoal)
                if rank < best_rank:
                    best_rank = rank
                    best = s

            if best is None:
                break

            path.append(best)
            prev = cur
            cur = best

        # if we couldn't build path, fallback greedy (won't freeze)
        if not path:
            x, y = self.start
            gx, gy = self.goal

            best = None
            best_score = 10**18

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map.width and 0 <= ny < self.map.height:
                    cand = (nx, ny)
                    if not is_valid_step(self.start, cand):
                        continue
                    c = float(self.known_costs[ny, nx])
                    if math.isinf(c):
                        continue

                    score = (abs(nx - gx) + abs(ny - gy)) * 10.0 + c
                    if score < best_score:
                        best_score = score
                        best = cand

            if best is None:
                return None

            self.planner.move_start(best)
            self.start = best
            return best

        # store cache and take first step now
        self._path_cache = path[1:]  # keep remaining
        first = path[0]

        self.planner.move_start(first)
        self.start = first
        return first


def navigate_multi_map(nav_graph: NavigationGraph,
                       start_map: str, start_pos: Tuple[int, int],
                       end_map: str, end_pos: Tuple[int, int]) -> Dict:
    result = {"success": False, "map_sequence": [], "checkpoints": {}, "paths": {}}

    seq = nav_graph.find_map_path(start_map, end_map)
    if not seq:
        return result

    result["success"] = True
    result["map_sequence"] = seq

    cur_pos = start_pos
    for i, m in enumerate(seq):
        is_last = (i == len(seq) - 1)
        target = end_pos if is_last else nav_graph.get_teleport_to(m, seq[i + 1])

        if target is None:
            result["success"] = False
            return result

        result["checkpoints"][m] = {"start": cur_pos, "goal": target}

        if not is_last:
            next_start = nav_graph.get_entry_teleport_from(m, seq[i + 1], goal_hint=target)
            cur_pos = next_start if next_start else target

    for m in seq:
        result["paths"][m] = []

    return result

