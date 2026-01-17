"""
Navigation Utilities for Wheelchair Accessibility Maps
Provides graph building and pathfinding functionality
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import IntEnum
import heapq

# Import from map_creator
from map_creator import MapCreator, MapPixel, FuncID


@dataclass
class TeleportInfo:
    """Information about a teleport (door/elevator/ramp/stair)"""
    position: Tuple[int, int]
    func_type: int  # FuncID value
    destinations: List[str]
    cost: int
    dept_id: int
    floor: int


@dataclass
class MapInfo:
    """Information about a loaded map"""
    map_id: str
    width: int
    height: int
    map_data: np.ndarray
    teleports: Dict[str, List[TeleportInfo]]  # destination -> list of teleport locations


class NavigationGraph:
    """Global navigation graph for multi-map pathfinding"""

    def __init__(self):
        self.maps: Dict[str, MapInfo] = {}
        self.graph: Dict[str, Set[str]] = {}  # map_id -> set of connected map_ids

    def load_maps(self, maps_directory: str):
        """Load all map files from directory and build navigation graph"""
        print(f"Loading maps from {maps_directory}...")

        if not os.path.exists(maps_directory):
            print(f"Error: Directory {maps_directory} does not exist")
            return

        map_files = [f for f in os.listdir(maps_directory) if f.endswith('.bin')]

        if not map_files:
            print(f"No .bin files found in {maps_directory}")
            return

        for map_file in map_files:
            filepath = os.path.join(maps_directory, map_file)
            map_id = self._load_map(filepath)
            if map_id:
                print(f"  ✓ Loaded: {map_id} ({map_file})")

        self._build_graph()
        print(f"\nGraph built with {len(self.maps)} maps")
        print(f"Total connections: {sum(len(v) for v in self.graph.values()) // 2}")

    def _load_map(self, filepath: str) -> Optional[str]:
        """Load a single map file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            map_id = data.get('map_id', os.path.basename(filepath)[:-4])
            width = data['width']
            height = data['height']

            # Reconstruct map_data
            map_data = np.array([[MapPixel.from_tuple(pixel_data)
                                  for pixel_data in row]
                                 for row in data['map_data']], dtype=object)

            # Find all teleports
            teleports = self._find_teleports(map_data, width, height)

            # Store map info
            self.maps[map_id] = MapInfo(
                map_id=map_id,
                width=width,
                height=height,
                map_data=map_data,
                teleports=teleports
            )

            return map_id

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def _find_teleports(self, map_data, width, height) -> Dict[str, List[TeleportInfo]]:
        """Find all teleports in a map and group by destination"""
        teleports: Dict[str, List[TeleportInfo]] = {}

        for y in range(height):
            for x in range(width):
                pixel = map_data[y][x]

                # Check if it's a teleport type
                if pixel.func_id in [FuncID.DOOR, FuncID.ELEVATOR,
                                     FuncID.RAMP, FuncID.STAIR]:
                    if pixel.identifier:  # Has destinations
                        teleport_info = TeleportInfo(
                            position=(x, y),
                            func_type=pixel.func_id,
                            destinations=pixel.identifier.copy(),
                            cost=pixel.cost,
                            dept_id=pixel.dept_id,
                            floor=pixel.floor
                        )

                        # Add to each destination's list
                        for dest in pixel.identifier:
                            if dest not in teleports:
                                teleports[dest] = []
                            teleports[dest].append(teleport_info)

        return teleports

    def _build_graph(self):
        """Build the navigation graph from loaded maps"""
        self.graph = {map_id: set() for map_id in self.maps}

        for map_id, map_info in self.maps.items():
            for dest_map_id in map_info.teleports.keys():
                if dest_map_id in self.maps:
                    self.graph[map_id].add(dest_map_id)
                    self.graph[dest_map_id].add(map_id)

    def find_map_path(self, start_map: str, end_map: str) -> Optional[List[str]]:
        """Find path between maps using BFS"""
        if start_map not in self.maps or end_map not in self.maps:
            return None

        if start_map == end_map:
            return [start_map]

        # BFS
        queue = [(start_map, [start_map])]
        visited = {start_map}

        while queue:
            current, path = queue.pop(0)

            for neighbor in self.graph.get(current, set()):
                if neighbor not in visited:
                    new_path = path + [neighbor]

                    if neighbor == end_map:
                        return new_path

                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def get_teleport_to(self, current_map: str, target_map: str) -> Optional[Tuple[int, int]]:
        """Get position of teleport from current_map to target_map"""
        if current_map not in self.maps:
            return None

        map_info = self.maps[current_map]
        teleports = map_info.teleports.get(target_map, [])

        if teleports:
            # Return position of first teleport
            return teleports[0].position

        return None

    def print_graph_info(self):
        """Print detailed graph information"""
        print("\n" + "=" * 60)
        print("NAVIGATION GRAPH INFORMATION")
        print("=" * 60)

        for map_id, map_info in sorted(self.maps.items()):
            print(f"\nMap: {map_id}")
            print(f"  Size: {map_info.width}x{map_info.height}")
            print(f"  Connections: {len(self.graph[map_id])}")

            if map_info.teleports:
                print(f"  Teleports:")
                for dest, teleports in sorted(map_info.teleports.items()):
                    print(f"    → {dest}: {len(teleports)} portal(s)")
                    for tp in teleports[:3]:  # Show first 3
                        func_name = FuncID(tp.func_type).name
                        print(f"       - {func_name} at {tp.position}")
                    if len(teleports) > 3:
                        print(f"       ... and {len(teleports) - 3} more")
            else:
                print(f"  No teleports (isolated map)")

    # ✅ NEW helper: choose correct entry teleport in a map
    def get_entry_teleport_from(self,
                                current_map: str,
                                next_map: str,
                                goal_hint: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """
        Find the teleport entry position INSIDE next_map that links back to current_map.

        Args:
            current_map: map we came from
            next_map: map we are entering
            goal_hint: position we want to be near (optional)

        Returns:
            (x, y) entry point inside next_map, or None if no candidate exists
        """
        if next_map not in self.maps:
            return None

        next_info = self.maps[next_map]
        candidates = next_info.teleports.get(current_map, [])

        if not candidates:
            return None

        if goal_hint is None:
            return candidates[0].position

        # Choose closest teleport to goal_hint (Manhattan distance)
        best = min(
            candidates,
            key=lambda tp: abs(tp.position[0] - goal_hint[0]) + abs(tp.position[1] - goal_hint[1])
        )
        return best.position


def find_path_in_map(map_data: np.ndarray, start: Tuple[int, int],
                     goal: Tuple[int, int], width: int, height: int) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding within a single map
    Returns list of (x, y) positions from start to goal
    """

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                pixel = map_data[ny][nx]
                # Skip obstacles (high cost) and stairs for wheelchair
                if pixel.cost < 900 and pixel.func_id != FuncID.STAIR:
                    neighbors.append(((nx, ny), pixel.cost))
        return neighbors

    # A* algorithm
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for neighbor, cost in get_neighbors(current):
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


def navigate_multi_map(nav_graph: NavigationGraph,
                       start_map: str, start_pos: Tuple[int, int],
                       end_map: str, end_pos: Tuple[int, int]) -> Dict:
    """
    Complete navigation across multiple maps
    Returns detailed navigation instructions
    """
    result = {
        'success': False,
        'map_sequence': [],
        'paths': {},
        'instructions': [],
        'total_cost': 0
    }

    # Find map sequence
    map_sequence = nav_graph.find_map_path(start_map, end_map)
    if not map_sequence:
        result['instructions'].append(f"No path found from {start_map} to {end_map}")
        return result

    result['map_sequence'] = map_sequence
    result['instructions'].append(f"Route: {' → '.join(map_sequence)}")

    # Navigate through each map
    current_pos = start_pos
    total_steps = 0

    for i, current_map in enumerate(map_sequence):
        map_info = nav_graph.maps[current_map]

        # Determine goal in this map
        if i == len(map_sequence) - 1:
            # Last map - go to final destination
            goal_pos = end_pos
        else:
            # Intermediate map - find teleport to next map
            next_map = map_sequence[i + 1]
            goal_pos = nav_graph.get_teleport_to(current_map, next_map)
            if not goal_pos:
                result['instructions'].append(f"Error: No teleport from {current_map} to {next_map}")
                return result

        # Find path within this map
        path = find_path_in_map(
            map_info.map_data,
            current_pos,
            goal_pos,
            map_info.width,
            map_info.height
        )

        if not path:
            result['instructions'].append(f"Error: No path in {current_map} from {current_pos} to {goal_pos}")
            return result

        result['paths'][current_map] = path
        total_steps += len(path)

        # Calculate cost
        path_cost = sum(map_info.map_data[y][x].cost for x, y in path)
        result['total_cost'] += path_cost

        # Add instruction
        if i == 0:
            result['instructions'].append(f"1. Start at {current_map} position {start_pos}")

        result['instructions'].append(f"{i + 2}. Navigate to {goal_pos} in {current_map} ({len(path)} steps)")

        # ✅ FIX: when moving to next map, compute REAL entry teleport in next map
        if i < len(map_sequence) - 1:
            next_map = map_sequence[i + 1]
            pixel = map_info.map_data[goal_pos[1]][goal_pos[0]]
            teleport_type = FuncID(pixel.func_id).name
            result['instructions'].append(f"   Use {teleport_type} to {next_map}")

            # choose a good hint inside next map:
            # if next map is final → hint = end_pos
            # else → hint near its teleport to the next-next map
            if i + 1 == len(map_sequence) - 1:
                goal_hint = end_pos
            else:
                after_next = map_sequence[i + 2]
                hint_tp = nav_graph.get_teleport_to(next_map, after_next)
                goal_hint = hint_tp if hint_tp else end_pos

            entry_pos = nav_graph.get_entry_teleport_from(
                current_map=current_map,
                next_map=next_map,
                goal_hint=goal_hint
            )

            if entry_pos is None:
                # fallback
                result['instructions'].append(
                    f"   Warning: No entry teleport in {next_map} linking back to {current_map}. Using same coordinate."
                )
                current_pos = goal_pos
            else:
                result['instructions'].append(f"   Enter {next_map} at {entry_pos}")
                current_pos = entry_pos

    result['success'] = True
    result['instructions'].append(f"\nTotal steps: {total_steps}")
    result['instructions'].append(f"Total cost: {result['total_cost']}")

    return result


def get_accessible_destinations(nav_graph: NavigationGraph,
                                from_map: str,
                                wheelchair_accessible_only: bool = True) -> List[str]:
    """
    Get all maps accessible from the given map
    If wheelchair_accessible_only, exclude paths that require stairs
    """
    if from_map not in nav_graph.maps:
        return []

    accessible = set()
    visited = {from_map}
    queue = [from_map]

    while queue:
        current = queue.pop(0)
        map_info = nav_graph.maps[current]

        for dest, teleports in map_info.teleports.items():
            if dest in nav_graph.maps and dest not in visited:
                # Check if wheelchair accessible
                if wheelchair_accessible_only:
                    has_accessible_route = any(
                        tp.func_type in [FuncID.DOOR, FuncID.ELEVATOR, FuncID.RAMP]
                        for tp in teleports
                    )
                    if not has_accessible_route:
                        continue

                accessible.add(dest)
                visited.add(dest)
                queue.append(dest)

    return sorted(accessible)


def validate_map_connections(nav_graph: NavigationGraph) -> Dict[str, List[str]]:
    """
    Validate that all teleport connections are bidirectional
    Returns dict of issues found
    """
    issues = {
        'one_way_connections': [],
        'missing_destinations': [],
        'isolated_maps': []
    }

    for map_id, map_info in nav_graph.maps.items():
        # Check if map is isolated
        if not map_info.teleports:
            issues['isolated_maps'].append(map_id)
            continue

        # Check each destination
        for dest in map_info.teleports.keys():
            # Check if destination map exists
            if dest not in nav_graph.maps:
                issues['missing_destinations'].append(f"{map_id} → {dest} (map not found)")
                continue

            # Check if destination has return path
            dest_map = nav_graph.maps[dest]
            if map_id not in dest_map.teleports:
                issues['one_way_connections'].append(f"{map_id} → {dest} (no return path)")

    return issues


if __name__ == "__main__":
    # Example usage
    print("Navigation Utilities Module")
    print("This module provides pathfinding and navigation functions.")
    print("\nExample usage:")
    print("""
    from nav_utils import NavigationGraph, navigate_multi_map

    # Load all maps
    nav_graph = NavigationGraph()
    nav_graph.load_maps('maps/')

    # Find path
    result = navigate_multi_map(
        nav_graph,
        start_map='campus',
        start_pos=(10, 10),
        end_map='cs_building_floor2',
        end_pos=(50, 50)
    )

    for instruction in result['instructions']:
        print(instruction)
    """)

