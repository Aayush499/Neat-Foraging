import pygame


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 1
        self.color = GREEN = (0, 255, 0)

    def draw(self, win):
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)