import pygame


class Obstacle:
    def __init__(self, startx, starty, endx, endy):
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy
        
        self.color = PURPLE = (128, 0, 128)


    def draw(self, win):
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)