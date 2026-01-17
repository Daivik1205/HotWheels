"""
Navigation Simulator - Visual wheelchair navigation with pygame
Shows real-time movement across multiple maps with status bar
"""

import pygame
import os
from typing import Tuple, Optional, List
from map_creator import MapCreator, MapPixel, FuncID
from nav_utils import NavigationGraph, navigate_multi_map, find_path_in_map


class NavigationSimulator:
    """Visual simulator for wheelchair navigation"""

    # Colors
    COLORS = {
        'background': (240, 240, 240),
        'grid': (200, 200, 200),
        'ui_bg': (50, 50, 50),
        'ui_text': (255, 255, 255),
        'wheelchair': (255, 0, 0),
        'path': (100, 200, 255),
        'start': (0, 255, 0),
        'goal': (255, 215, 0),
        FuncID.EMPTY: (255, 255, 255),
        FuncID.WALKABLE: (200, 255, 200),
        FuncID.RAMP: (150, 200, 255),
        FuncID.DOOR: (255, 200, 150),
        FuncID.ELEVATOR: (200, 150, 255),
        FuncID.STAIR: (255, 150, 150),
        FuncID.OBSTACLE: (100, 100, 100),
    }

    def __init__(self, nav_graph: NavigationGraph, window_width=1200, window_height=800):
        pygame.init()

        self.nav_graph = nav_graph
        self.window_width = window_width
        self.window_height = window_height

        # Window setup
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Wheelchair Navigation Simulator")

        # UI dimensions
        self.statusbar_height = 80
        self.canvas_height = window_height - self.statusbar_height

        # Current state
        self.current_map = None
        self.current_map_data = None
        self.map_width = 0
        self.map_height = 0

        # Navigation state
        self.wheelchair_pos = None
        self.goal_pos = None
        self.current_path = []
        self.path_index = 0
        self.map_sequence = []
        self.current_map_index = 0
        self.full_navigation = None
        self.teleport_transitions = []  # Store teleport entry/exit points

        # Animation
        self.animation_speed = 5  # frames per step
        self.animation_counter = 0
        self.is_moving = False
        self.total_steps = 0
        self.total_cost = 0

        # View properties
        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 10.0
        self.offset_x = 0
        self.offset_y = 0

        # Color array cache
        self.color_array = None

        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)

        # Clock
        self.clock = pygame.time.Clock()
        self.running = True

        # Input mode
        self.input_mode = None  # 'start', 'goal', 'switch_map', or None
        self.pending_start = None
        self.pending_goal = None
        self.pending_start_map = None
        self.pending_goal_map = None
        self.map_input_buffer = ""  # For map switching input

    def load_map(self, map_id: str):
        """Load a specific map for display"""
        if map_id not in self.nav_graph.maps:
            print(f"Error: Map {map_id} not found")
            return False

        map_info = self.nav_graph.maps[map_id]
        self.current_map = map_id
        self.current_map_data = map_info.map_data
        self.map_width = map_info.width
        self.map_height = map_info.height

        # Build color array
        self.color_array = self._build_color_array()

        # Center view
        self.offset_x = (self.window_width - self.map_width * self.zoom) / 2
        self.offset_y = (self.canvas_height - self.map_height * self.zoom) / 2

        pygame.display.set_caption(f"Navigation Simulator - {map_id}")

        return True

    def _build_color_array(self):
        """Build color array for fast rendering"""
        color_array = []
        for y in range(self.map_height):
            row = []
            for x in range(self.map_width):
                pixel = self.current_map_data[y][x]
                color = self.COLORS.get(pixel.func_id, self.COLORS[FuncID.EMPTY])
                row.append(color)
            color_array.append(row)
        return color_array

    def world_to_screen(self, wx, wy):
        """Convert world coordinates to screen coordinates"""
        sx = (wx * self.zoom) + self.offset_x
        sy = (wy * self.zoom) + self.offset_y
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy):
        """Convert screen coordinates to world coordinates"""
        wx = (sx - self.offset_x) / self.zoom
        wy = (sy - self.offset_y) / self.zoom
        return int(wx), int(wy)

    def screen_to_map(self, sx, sy):
        """Convert screen coordinates to map coordinates"""
        wx, wy = self.screen_to_world(sx, sy)
        if 0 <= wx < self.map_width and 0 <= wy < self.map_height:
            return wx, wy
        return None

    def start_navigation(self, start_map: str, start_pos: Tuple[int, int],
                         end_map: str, end_pos: Tuple[int, int]):
        """Start navigation from start to end"""
        print(f"\nStarting navigation: {start_map}{start_pos} → {end_map}{end_pos}")

        # Get full navigation plan
        self.full_navigation = navigate_multi_map(
            self.nav_graph,
            start_map, start_pos,
            end_map, end_pos
        )

        if not self.full_navigation['success']:
            print("Navigation failed!")
            for instruction in self.full_navigation['instructions']:
                print(instruction)
            return False

        print("\nNavigation plan:")
        for instruction in self.full_navigation['instructions']:
            print(instruction)

        # Set up state
        self.map_sequence = self.full_navigation['map_sequence']
        self.current_map_index = 0
        self.wheelchair_pos = start_pos
        self.goal_pos = end_pos
        self.total_steps = 0
        self.total_cost = 0

        # Build teleport transition data
        self._build_teleport_transitions()

        # Load first map
        self.load_map(self.map_sequence[0])

        # Get path for first map
        self._load_current_map_path()

        self.is_moving = True
        return True

    def _build_teleport_transitions(self):
        """Build data structure for teleport entry/exit points"""
        self.teleport_transitions = []

        for i in range(len(self.map_sequence) - 1):
            current_map_id = self.map_sequence[i]
            next_map_id = self.map_sequence[i + 1]

            # Get exit point from current map (last point in path)
            current_path = self.full_navigation['paths'][current_map_id]
            exit_point = current_path[-1]

            # Find teleport type at exit
            current_map = self.nav_graph.maps[current_map_id]
            exit_pixel = current_map.map_data[exit_point[1]][exit_point[0]]

            # Entry point in next map = first position in next map path
            next_path = self.full_navigation['paths'][next_map_id]
            entry_point = next_path[0]

            self.teleport_transitions.append({
                'from_map': current_map_id,
                'to_map': next_map_id,
                'exit_point': exit_point,
                'entry_point': entry_point,
                'teleport_type': FuncID(exit_pixel.func_id).name
            })

            print(f"Teleport {i + 1}: {current_map_id}{exit_point} → {next_map_id}{entry_point} via {FuncID(exit_pixel.func_id).name}")

    def _load_current_map_path(self):
        """Load path for current map in sequence"""
        current_map = self.map_sequence[self.current_map_index]

        if current_map in self.full_navigation['paths']:
            self.current_path = self.full_navigation['paths'][current_map]
            self.path_index = 0
            print(f"Loaded path for {current_map}: {len(self.current_path)} steps")
        else:
            print(f"Error: No path for {current_map}")
            self.current_path = []

    def update(self):
        """Update navigation state"""
        if not self.is_moving or not self.current_path:
            return

        self.animation_counter += 1

        if self.animation_counter >= self.animation_speed:
            self.animation_counter = 0

            # Move to next position
            if self.path_index < len(self.current_path):
                self.wheelchair_pos = self.current_path[self.path_index]

                # Calculate cost
                x, y = self.wheelchair_pos
                pixel = self.current_map_data[y][x]
                self.total_cost += pixel.cost
                self.total_steps += 1

                self.path_index += 1

                # End of current path
                if self.path_index >= len(self.current_path):
                    if self.current_map_index < len(self.map_sequence) - 1:
                        transition = self.teleport_transitions[self.current_map_index]

                        print(f"\nReached teleport in {transition['from_map']} at {transition['exit_point']}")
                        print(f"Using {transition['teleport_type']} to teleport to {transition['to_map']}")

                        # Move to next map
                        self.current_map_index += 1
                        next_map = self.map_sequence[self.current_map_index]

                        print(f"Entering {next_map} at {transition['entry_point']}")

                        # Load next map + next path
                        self.load_map(next_map)
                        self._load_current_map_path()

                        if self.current_path:
                            expected_entry = transition['entry_point']
                            path_start = self.current_path[0]

                            # ✅ Always start exactly at the teleport entry point
                            self.wheelchair_pos = expected_entry

                            # If the path doesn't start from expected_entry, stitch a small connector path
                            if expected_entry != path_start:
                                print("  ⚠ Next-map path does not start at teleport entry point.")
                                print("  → Stitching connector path entry → path_start")

                                connector = find_path_in_map(
                                    self.current_map_data,
                                    expected_entry,
                                    path_start,
                                    self.map_width,
                                    self.map_height
                                )

                                if connector:
                                    # avoid duplicating path_start
                                    if connector[-1] == path_start:
                                        connector = connector[:-1]
                                    self.current_path = connector + self.current_path
                                    print(f"  ✓ Added {len(connector)} connector steps")
                                else:
                                    print("  ✗ Failed to build connector path; forcing path_start")
                                    self.wheelchair_pos = path_start

                            # We are at index 0 now
                            self.path_index = 1
                        else:
                            print("Error: No path in new map!")

                    else:
                        print(f"\n{'=' * 60}")
                        print(f"✓ DESTINATION REACHED!")
                        print(f"{'=' * 60}")
                        print(f"Total steps: {self.total_steps}")
                        print(f"Total cost: {self.total_cost}")
                        print(f"Maps traversed: {len(self.map_sequence)}")
                        print(f"Route: {' → '.join(self.map_sequence)}")
                        print(f"{'=' * 60}")
                        self.is_moving = False

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_moving = not self.is_moving
                    print("Paused" if not self.is_moving else "Resumed")

                elif event.key == pygame.K_r:
                    if self.full_navigation:
                        start_map = self.map_sequence[0]
                        start_pos = self.full_navigation['paths'][start_map][0]
                        end_map = self.map_sequence[-1]
                        end_pos = self.full_navigation['paths'][end_map][-1]
                        self.start_navigation(start_map, start_pos, end_map, end_pos)

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.animation_speed = max(1, self.animation_speed - 1)
                    print(f"Speed: {self.animation_speed} frames/step")

                elif event.key == pygame.K_MINUS:
                    self.animation_speed = min(30, self.animation_speed + 1)
                    print(f"Speed: {self.animation_speed} frames/step")

                elif event.key == pygame.K_s:
                    print("\nClick on map to set START position")
                    self.input_mode = 'start'

                elif event.key == pygame.K_g:
                    print("\nClick on map to set GOAL position")
                    self.input_mode = 'goal'

                elif event.key == pygame.K_n:
                    if self.pending_start and self.pending_goal:
                        self.start_navigation(
                            self.pending_start_map, self.pending_start,
                            self.pending_goal_map, self.pending_goal
                        )
                    else:
                        print("\nSet start (S) and goal (G) positions first, then press N")

                elif event.key == pygame.K_m:
                    print("\nAvailable maps:")
                    map_list = sorted(self.nav_graph.maps.keys())
                    for i, map_id in enumerate(map_list):
                        print(f"  {i + 1}. {map_id}")
                    self.input_mode = 'switch_map'
                    print("Type number and press ENTER to switch, or ESC to cancel")

                elif event.key == pygame.K_ESCAPE:
                    self.input_mode = None
                    print("\nInput cancelled")

                elif self.input_mode == 'switch_map':
                    if event.key == pygame.K_RETURN:
                        if hasattr(self, 'map_input_buffer') and self.map_input_buffer:
                            try:
                                map_list = sorted(self.nav_graph.maps.keys())
                                map_idx = int(self.map_input_buffer) - 1
                                if 0 <= map_idx < len(map_list):
                                    selected_map = map_list[map_idx]
                                    self.load_map(selected_map)
                                    print(f"Switched to {selected_map}")
                                else:
                                    print(f"Invalid map number: {self.map_input_buffer}")
                            except ValueError:
                                print(f"Invalid input: {self.map_input_buffer}")
                            self.map_input_buffer = ""
                            self.input_mode = None
                    elif event.key == pygame.K_BACKSPACE:
                        if hasattr(self, 'map_input_buffer'):
                            self.map_input_buffer = self.map_input_buffer[:-1]
                    elif event.unicode.isdigit():
                        if not hasattr(self, 'map_input_buffer'):
                            self.map_input_buffer = ""
                        self.map_input_buffer += event.unicode
                        print(f"Input: {self.map_input_buffer}")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if event.button == 1:  # Left click
                    if self.input_mode:
                        map_pos = self.screen_to_map(mouse_pos[0], mouse_pos[1])
                        if map_pos:
                            if self.input_mode == 'start':
                                self.pending_start = map_pos
                                self.pending_start_map = self.current_map
                                print(f"Start set to {self.current_map}:{map_pos}")
                                self.input_mode = None
                            elif self.input_mode == 'goal':
                                self.pending_goal = map_pos
                                self.pending_goal_map = self.current_map
                                print(f"Goal set to {self.current_map}:{map_pos}")
                                self.input_mode = None

                            if self.pending_start and self.pending_goal:
                                print("\nPress N to start navigation")

            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[1] < self.canvas_height:
                    if event.y > 0:
                        self.zoom = min(self.zoom * 1.1, self.max_zoom)
                    else:
                        self.zoom = max(self.zoom / 1.1, self.min_zoom)

                    wx, wy = self.screen_to_world(mouse_pos[0], mouse_pos[1])
                    self.offset_x = mouse_pos[0] - wx * self.zoom
                    self.offset_y = mouse_pos[1] - wy * self.zoom

    def render_map(self):
        """Render the current map"""
        self.screen.fill(self.COLORS['background'])

        if not self.current_map_data.all():
            return

        top_left = self.screen_to_world(0, 0)
        bottom_right = self.screen_to_world(self.window_width, self.canvas_height)

        min_x = max(0, int(top_left[0]))
        min_y = max(0, int(top_left[1]))
        max_x = min(self.map_width, int(bottom_right[0]) + 2)
        max_y = min(self.map_height, int(bottom_right[1]) + 2)

        pixel_size = max(1, int(self.zoom))

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                color = self.color_array[y][x]
                sx, sy = self.world_to_screen(x, y)
                pygame.draw.rect(self.screen, color, (sx, sy, pixel_size, pixel_size))

        # Draw path
        if self.current_path:
            for i, (px, py) in enumerate(self.current_path):
                if i < self.path_index:
                    color = (180, 220, 255)
                else:
                    color = self.COLORS['path']

                sx, sy = self.world_to_screen(px, py)
                pygame.draw.circle(
                    self.screen, color,
                    (sx + pixel_size // 2, sy + pixel_size // 2),
                    max(2, pixel_size // 3)
                )

        # Draw start position (if pending)
        if self.pending_start and self.pending_start_map == self.current_map:
            sx, sy = self.world_to_screen(self.pending_start[0], self.pending_start[1])
            pygame.draw.circle(
                self.screen, self.COLORS['start'],
                (sx + pixel_size // 2, sy + pixel_size // 2),
                max(5, pixel_size // 2)
            )

        # Draw goal position (if pending)
        if self.pending_goal and self.pending_goal_map == self.current_map:
            sx, sy = self.world_to_screen(self.pending_goal[0], self.pending_goal[1])
            pygame.draw.circle(
                self.screen, self.COLORS['goal'],
                (sx + pixel_size // 2, sy + pixel_size // 2),
                max(5, pixel_size // 2)
            )

        # Draw wheelchair
        if self.wheelchair_pos:
            wx, wy = self.wheelchair_pos
            sx, sy = self.world_to_screen(wx, wy)
            radius = max(6, pixel_size // 2 + 2)
            pygame.draw.circle(
                self.screen, self.COLORS['wheelchair'],
                (sx + pixel_size // 2, sy + pixel_size // 2),
                radius
            )
            pygame.draw.circle(
                self.screen, (255, 255, 255),
                (sx + pixel_size // 2, sy + pixel_size // 2),
                radius, 2
            )

    def render_status_bar(self):
        """Render status bar with information"""
        status_rect = pygame.Rect(0, self.canvas_height,
                                  self.window_width, self.statusbar_height)
        pygame.draw.rect(self.screen, self.COLORS['ui_bg'], status_rect)

        y_offset = self.canvas_height + 10

        if self.current_map and self.wheelchair_pos:
            text = f"Map: {self.current_map} | Position: {self.wheelchair_pos}"
            surf = self.font_medium.render(text, True, self.COLORS['ui_text'])
            self.screen.blit(surf, (10, y_offset))
            y_offset += 25

        if self.map_sequence:
            progress_text = f"Route: {' → '.join(self.map_sequence)}"
            surf = self.font_small.render(progress_text, True, self.COLORS['ui_text'])
            self.screen.blit(surf, (10, y_offset))
            y_offset += 20

            if self.current_path:
                progress = f"Map {self.current_map_index + 1}/{len(self.map_sequence)} | " \
                           f"Steps: {self.path_index}/{len(self.current_path)} | " \
                           f"Total: {self.total_steps} steps, Cost: {self.total_cost}"
                surf = self.font_small.render(progress, True, self.COLORS['ui_text'])
                self.screen.blit(surf, (10, y_offset))
                y_offset += 20

        controls = "SPACE: Pause/Resume | +/-: Speed | R: Reset | S: Set Start | G: Set Goal | N: Navigate | M: Switch Map"
        surf = self.font_small.render(controls, True, (150, 150, 150))
        self.screen.blit(surf, (10, self.window_height - 20))

        status = "MOVING" if self.is_moving else "PAUSED"
        if not self.current_path:
            status = "IDLE"

        status_surf = self.font_large.render(
            status, True,
            (0, 255, 0) if self.is_moving else (255, 100, 100)
        )
        status_rect = status_surf.get_rect(right=self.window_width - 20, top=self.canvas_height + 10)
        self.screen.blit(status_surf, status_rect)

    def run(self):
        """Main simulation loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render_map()
            self.render_status_bar()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def run_simulator(maps_dir='maps', start_map=None, start_pos=None,
                  end_map=None, end_pos=None):
    """
    Run the navigation simulator
    """
    print("Loading maps...")
    nav_graph = NavigationGraph()
    nav_graph.load_maps(maps_dir)

    if not nav_graph.maps:
        print(f"Error: No maps found in {maps_dir}")
        return

    print(f"Loaded {len(nav_graph.maps)} maps")

    simulator = NavigationSimulator(nav_graph)

    first_map = start_map if start_map else list(nav_graph.maps.keys())[0]
    simulator.load_map(first_map)

    if start_map and start_pos and end_map and end_pos:
        simulator.start_navigation(start_map, start_pos, end_map, end_pos)
    else:
        print("\nSimulator started in IDLE mode")
        print("Press S to set start position")
        print("Press G to set goal position")
        print("Press N to start navigation")
        print("Press M to switch maps")

    simulator.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if len(sys.argv) >= 9:
            run_simulator(
                maps_dir=sys.argv[1],
                start_map=sys.argv[2],
                start_pos=(int(sys.argv[3]), int(sys.argv[4])),
                end_map=sys.argv[5],
                end_pos=(int(sys.argv[6]), int(sys.argv[7]))
            )
        else:
            run_simulator(maps_dir=sys.argv[1])
    else:
        print("Usage:")
        print("  python simulator.py [maps_dir]")
        print("  python simulator.py [maps_dir] [start_map] [start_x] [start_y] [end_map] [end_x] [end_y]")
        print("\nRunning with default 'maps' directory...")
        run_simulator('maps')

