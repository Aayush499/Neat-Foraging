import pygame

class Pheromone:
    def __init__(self, x, y, strength):
        self.x = x
        self.y = y
        self.strength = strength
        self.radius = 3
        self.color = (255, 255, 0)

    def draw(self, win):
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)