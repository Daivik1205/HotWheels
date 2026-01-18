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
        return min(candidates, key=lambda tp: abs(tp.position[0] - goal_hint[0]) + abs(tp.position[1] - goal_hint[1])).position


class DynamicLocalPlanner:
    """
    D* Lite local planner with limited sight.
    Unknown cells are assumed free (cost=1) until discovered.
    """

    def __init__(self, map_info, start, goal):
        self.map = map_info
        self.start = start
        self.goal = goal

        self.gt_costs = self._build_ground_truth_cost_grid()
        self.known_costs = np.ones_like(self.gt_costs, dtype=np.float32)

        self._sanitize_start_goal()

        self.planner = DStarLite(self.start, self.goal, self.known_costs)
        self.planner.compute(max_steps=300)

        sx, sy = self.start
        gx, gy = self.goal 
        print("gt start cost", self.gt_costs[sy, sx], "gt goal cost", self.gt_costs[gy, gx])

    def _build_ground_truth_cost_grid(self):
        w, h = self.map.width, self.map.height
        grid = np.ones((h, w), dtype=np.float32)  # [H,W]

        for y in range(h):
            for x in range(w):
                p = self.map.map_data[y][x]
                if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
                    grid[y, x] = np.inf
                else:
                    grid[y, x] = max(1.0, float(p.cost))
        return grid

    def _sanitize_start_goal(self):
        def free(cell):
            x, y = cell
            return not math.isinf(float(self.gt_costs[y, x]))

        # If start/goal blocked -> fail fast (no movement)
        if not free(self.start) or not free(self.goal):
            # keep as-is; planner will return None safely
            return

    def sense_and_update(self, visible_cells):
        # Live refresh ground truth from map_data so newly added obstacles matter
        for (x, y) in visible_cells:
            p = self.map.map_data[y][x]
            true_cost = np.inf if (p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999) else max(1.0, float(p.cost))

            known = float(self.known_costs[y, x])
            if (math.isinf(true_cost) and not math.isinf(known)) or (not math.isinf(true_cost) and abs(true_cost - known) > 1e-6):
                self.known_costs[y, x] = true_cost
                self.planner.update_cell_cost((x, y), true_cost)

    def step(self, compute_budget=300):
        nxt = self.planner.next(compute_budget=compute_budget)
        if nxt is not None:
            return nxt
        
        x, y = self.start
        gx, gy = self.goal

        best = None 
        best_h = 10**9

        for dx, dy in [(1,0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]:
            nx, ny = x+dx, y+dy 
            if 0 <= nx < self.map.width and 0 <= self.map.height:
                if not math.isinf(float(self.known_costs[ny, nx])):
                    h = abs(nx-gx) + abs(ny-gy)
                    if h < best_h:
                        best_h = h 
                        best = (nx, ny)
        if best is not None:
            self.planner.move_start(best)
            self.start = best 
            return best
        return None

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

