import pygame
import math

class Nest:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 20
        self.color = PINK = (255, 192, 203)

    def draw(self, win):
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
