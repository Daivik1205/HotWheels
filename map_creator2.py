import pygame
import pickle
from dataclasses import dataclass, field

# ---------------------------------
# constants
# ---------------------------------
TILE = 24
GRID_W, GRID_H = 30, 20

EMPTY = 0
WALK = 1
RAMP = 2
DOOR = 3
ELEVATOR = 4
OBSTACLE = 6

FILE_NAME = "map.bin"


# ---------------------------------
# MapPixel
# ---------------------------------
@dataclass
class MapPixel:
    cost: int = 0
    dept_id: int = 0
    floor: int = 0
    func_id: int = 0
    identifier: list = field(default_factory=list)


# ---------------------------------
# TextBox widget
# ---------------------------------
class TextBox:
    def __init__(self, rect, font):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.active = False
        self.text = ""
        self.prompt = "destinations (comma separated)"

    def handle(self, e):
        if not self.active:
            return None

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                value = self.text.strip()
                self.text = ""
                self.active = False
                return value

            elif e.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]

            else:
                self.text += e.unicode

        return None

    def draw(self, screen):
        pygame.draw.rect(screen, (30, 30, 30), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        t = self.text if self.text else self.prompt
        surf = self.font.render(t, True, (230, 230, 230))
        screen.blit(surf, (self.rect.x + 6, self.rect.y + 6))


# ---------------------------------
# helpers
# ---------------------------------
def set_dest(pixel, text_value, func_id):
    pixel.func_id = func_id
    pixel.identifier = [v.strip() for v in text_value.split(",") if v.strip()]


def save_map(grid):
    data = {
        "width": GRID_W,
        "height": GRID_H,
        "map_data": grid,
    }
    with open(FILE_NAME, "wb") as f:
        pickle.dump(data, f)
    print("Saved:", FILE_NAME)


# ---------------------------------
# main editor
# ---------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * TILE, GRID_H * TILE + 60))
    pygame.display.set_caption("Map Creator")

    font = pygame.font.SysFont("Arial", 18)
    textbox = TextBox((10, GRID_H * TILE + 10, 560, 40))

    grid = [[MapPixel() for _ in range(GRID_W)] for _ in range(GRID_H)]

    mode = DOOR
    selected = None
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            submitted = textbox.handle(e)
            if submitted and selected:
                x, y = selected
                set_dest(grid[y][x], submitted, mode)
                print("Updated:", grid[y][x].identifier)

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_d:
                    mode = DOOR
                elif e.key == pygame.K_e:
                    mode = ELEVATOR
                elif e.key == pygame.K_r:
                    mode = RAMP
                elif e.key == pygame.K_s:
                    save_map(grid)

            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if my < GRID_H * TILE:
                    x = mx // TILE
                    y = my // TILE
                    selected = (x, y)
                    textbox.active = True
                    textbox.text = ""

        # draw map
        screen.fill((18, 18, 18))

        for y in range(GRID_H):
            for x in range(GRID_W):
                p = grid[y][x]

                color = (70, 70, 70)
                if p.func_id == DOOR:
                    color = (255, 200, 120)
                elif p.func_id == ELEVATOR:
                    color = (200, 150, 255)
                elif p.func_id == RAMP:
                    color = (150, 200, 255)

                rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (40, 40, 40), rect, 1)

        textbox.draw(screen)

        info = font.render(
            f"Mode: {('DOOR','WALK','RAMP','DOOR','ELEVATOR')[mode]} | D/E/R change mode | S save",
            True,
            (230, 230, 230),
        )
        screen.blit(info, (600, GRID_H * TILE + 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

