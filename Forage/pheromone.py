import pygame

class Pheromone:
    def __init__(self, x, y, strength):
        self.x = x
        self.y = y
        self.strength = strength
        self.radius = 3
        self.color = (255, 255, 0)

    def draw(self, win):
        x, y, strength = self.x, self.y, self.strength
        if strength > 0:
            alpha = int(255 * strength)
            color = (255, 255, 0, alpha)
            pygame.draw.circle(win, self.color, (x, y), self.radius)