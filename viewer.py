import pygame

TILE = 20
ZOOM = 1.2


class Viewer:
    def __init__(self, w, h):
        pygame.init()
        self.screen = pygame.display.set_mode((900, 700), pygame.RESIZABLE)
        self.offset = [0, 0]
        self.zoom = 1.0
        self.drag = False
        self.last = (0, 0)
        self.font = pygame.font.SysFont("Arial", 14)
        self.w = w
        self.h = h

    def world(self, x, y):
        return (int(x*TILE*self.zoom + self.offset[0]),
                int(y*TILE*self.zoom + self.offset[1]))

    def draw(self, grid, wc, fov, path):
        self.screen.fill((25,25,25))

        for y,r in enumerate(grid):
            for x,v in enumerate(r):
                rect = (*self.world(x,y), int(TILE*self.zoom), int(TILE*self.zoom))
                color = (240,240,240) if v == 0 else (90,90,90)
                pygame.draw.rect(self.screen, color, rect)

        for cx,cy in fov:
            pygame.draw.rect(self.screen, (80,150,255),
                             (*self.world(cx,cy), int(TILE*self.zoom), int(TILE*self.zoom)), 2)

        for px,py in path:
            pygame.draw.circle(self.screen, (255,0,0),
                               self.world(px+0.5, py+0.5), 4)

        wx, wy = wc.pos
        pygame.draw.circle(self.screen, (0,255,0),
                           self.world(wx+0.5, wy+0.5), 6)

        pygame.display.flip()

    def events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self.drag = True; self.last = e.pos
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.drag = False
            if e.type == pygame.MOUSEWHEEL:
                self.zoom *= (ZOOM if e.y>0 else 1/ZOOM)
            if e.type == pygame.MOUSEMOTION and self.drag:
                dx,dy = e.pos[0]-self.last[0], e.pos[1]-self.last[1]
                self.offset[0]+=dx; self.offset[1]+=dy; self.last=e.pos
        return True

